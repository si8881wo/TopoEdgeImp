module EdgeImpV8 # v8
########################################################################################
# packages needed
using Distributed # for paralellization
using LinearAlgebra, DataFrames, CSV
########################################################################################
# functions to export
export thomas # thomas alg. not touching input vectors
export thomas! # thomas alg. mutating input vectors
export gR_V # unperturbed Green's function in position representation between two imps
export g # unperturbed Green's function in position representation between all imps
export Mimpsingle # single impurity potential matrix/FCs
export imps # create random impurity distribution and most centered imp index
export LHS # set up left hand sinde of matrix eq
export Gndirect # solve matrix eq in position rep.
export DOSfc_from_G_N_full # extract DOS from result of Gndirect
export G_lattice_dos
export T_DOS
export Tk_DOS
export write_to_df
export read_from_csv
export write_to_df_full
export read_from_csv_full
########################################################################################

# Inputs and typical values
# LolB=8.0 # system length parameter
# omega=1.0 # driving frequency in units of the gap
# nmax=3 # index of highest Fourier components -> 2nmax+1 total FCs
# Nruns=100 # No. of impurity averaging runs
# V=0.0 # potential strength of the imps relative to M
# M=0.1 # magnetic strength of the imps in units of \hbar\nu_F
# Emax=5.0 # Energy around gap center
# Enum=101 # No. of energy points

########################################################
######## Some general functions and definitions ########
########################################################

# Pauli matricies
sigma0=Matrix{ComplexF64}(I, 2, 2)
sigmax=Matrix{ComplexF64}([0.0 1.0; 1.0 0.0])
sigmay=Matrix{ComplexF64}([0.0 -im; im 0.0])
sigmaz=Matrix{ComplexF64}([1.0 0.0; 0.0 -1.0])
# vector containing the pauli matrices
sigma=[[sigmax];[sigmay];[sigmaz]]

##### Thomas algorithm to efficiently solve the block-tridiagonal matrix equation #####
# A,B,C Vectors containing the lower/upper/diagonal blocks, !Careful: index is the row of the matrix so A[1]=B[N]=0
# F Vector containing the blocks of the RHS of the matrix eq.
# n block size (corresponds to no. of imps.)
# N number of blocks/FCs

function thomas(A,B,C,F,n,N)
    
    # set up some auxiliary vectors of matrices, probably dont have to store everythin/can overwrite the old vectors
    a=[complex(zeros(n,n)) for i=1:N]
    b=[complex(zeros(n,n)) for i=1:N]
    g=[complex(zeros(n,n)) for i=1:N]
    
    # initial values
    g[1]=\(C[1],I)
    a[1]=g[1]*B[1]
    b[1]=g[1]*F[1]
    
    # forward sweep
    for i=2:N
        g[i]=\(C[i]-A[i]*a[i-1],I)
        a[i]=g[i]*B[i]
        b[i]=g[i]*(F[i]-A[i]*b[i-1])
    end
    
    # inital valueize and initial value
    G=[complex(zeros(n,n)) for i=1:N]
    G[N]=b[N]
    # backward sweep
    for i=N-1:-1:1
        G[i]=b[i]-a[i]*G[i+1]
    end
    
    return G
end

# more memory efficient (also seems slightly faster) version
# !CAREFUL! mutates the vectors that contain the blocks of the matrix (A,B,C)
function thomas!(A,B,C,F,n,N) 
        
    # initial values
    C[1]=\(C[1],I)
    A[1]=C[1]*B[1]
    B[1]=C[1]*F[1]
    
    # forward sweep
    for i=2:N
        Ai,Bi=A[i],B[i]
        C[i]=\(C[i]-Ai*A[i-1],I)
        A[i]=C[i]*Bi
        B[i]=C[i]*(F[i]-Ai*B[i-1])
    end
    
    # inital valueize and initial value
    C[N]=B[N]
    # backward sweep
    for i=N-1:-1:1
        C[i]=B[i]-A[i]*C[i+1]
    end
    
    return C
end
##############################

########################################################
########################################################
########################################################

########################################################
########  System specific functions for set up  ########
########################################################

# free GF for impurity system
##### unperturbed Green's function sub matrices in position representation #####
function gR_V(E,u,up,LolB,V,Xvec)
    ########################################################################################
    # E: Energy scaled in units of Delta                                                   #
    # u=x/L impurity position relative to L                                                #
    # up=x'/L impurity position relative to L                                              #
    # LolB: L/l_B ratio system length/decay length, controls magnetic impurity strength    #
    # V: potential part of the impurity potential relative to M                            #
    # Xvec: impurity vector containing impurity positions relative to L                    #
    ########################################################################################
    #initialize empty matrix
    gR=zeros(ComplexF64,2,2)
    # Extract number of impurities
    N=size(Xvec)[1]
    # Heaviside function
    epsilon=1e-16 # Parameter controlling the Heaviside "sharpness" or width
    HS=0.50*(1.0+(2.0*atan((u-up)/epsilon)/pi))  # Smooth Heaviside of width epsilon
    # free impurity g
    gR[1,1]=-im*exp( im*E*LolB*(u-up))*HS
    gR[2,2]=-im*exp(-im*E*LolB*(u-up))*(1.0-HS)
    # Calculate phase factor due do potential part
    Theta =0.0
    Thetap=0.0
    for n=1:N
        Theta =Theta +V*(LolB/N)*0.50*(1.0+(2.0*atan((u-Xvec[n])/epsilon)/pi))
        Thetap=Thetap+V*(LolB/N)*0.50*(1.0+(2.0*atan((up-Xvec[n])/epsilon)/pi))
    end
    # in matrices
    expTheta=[exp(-im*Theta) 0.0; 0.0 exp(im*Theta)]
    expThetap=[exp(im*Thetap) 0.0; 0.0 exp(-im*Thetap)]
    # return g0 with proper phase factors
    return expTheta*gR*expThetap
end
##### unperturbed Green's function in position representation, fill with submatrices #####
function g(E,Xvec,LolB,V)
    ########################################################################################
    # E: Energy scaled in units of Delta                                                   #
    # LolB: L/l_B ratio system length/decay length, controls magnetic impurity strength    #
    # V: potential part of the impurity potential relative to M                            #
    # Xvec: impurity vector containing impurity positions relative to L                    #
    ########################################################################################
	Nimp=length(Xvec) # number of impurities
	g0=Matrix{ComplexF64}(undef,2*Nimp, 2*Nimp) # initialize empty matrix
    # fill up with the submatrices
	for i=1:Nimp
		xi=Xvec[i]
        for j=1:Nimp
            xj=Xvec[j]
            g0[2*i-1:2*i,2*j-1:2*j]=gR_V(E,xi,xj,LolB,V,Xvec)
        end
	end
    return g0
end

##### creating impurity potentials and impurity distributions #####
# create 2x2 single impurity matrix fourier coefficients for rotation in x-z-plane and static contribution of Ms in y-direction
# M total impurity strength, Ms ratio M in static part
function Mimpsingle(M,Ms)
    M0=M*Ms*sigmay
    Mp=sqrt(1-Ms*Ms)*M/(2*im)*(sigmax+im*sigmaz)
    Mm=Mp'
    return (M0,Mp,Mm)
end

# create random impurity distribution
function imps(Nimp)
   # create random impurity positions
    #srand(2) # Set the random seed
    Xvec=rand(Float64,Nimp)
    sort!(Xvec)  # sort the impurity position
    minimaldistance=minimum(Xvec[2:Nimp]-Xvec[1:Nimp-1])
    # Make sure impurities are reasonably far apart
    while (minimaldistance<=10e-6)
	Xvec=rand(Float64,Nimp)
    	sort!(Xvec)  # sort the impurity position
    	minimaldistance=minimum(Xvec[2:Nimp]-Xvec[1:Nimp-1])
    end
    # find position and index of impurity closest to the center
    cent=findmin(abs.(Xvec.-0.5))[2]
    return Xvec,cent
end
##############################

## set up matrix for matrix eq./LHS ##
function LHS(E,V,M0,Mp,Mm,Xvec,Omega,nmax,Nimp,gfsize,LolB=8)
    
    # unit matrices
    Id_Nimp=Matrix{ComplexF64}(I,Nimp,Nimp) # unit matrix
    Id_2Nimp=Matrix{ComplexF64}(I,2*Nimp,2*Nimp) # unit matrix
        
    # big impurity matrices
    V0,Vp,Vm=map(x->kron(Id_Nimp,x),(M0,Mp,Mm))
    
    #initialize vectors to fill with the lower/upper/diagonal(-/+/0 components) blocks of the matrix eq, blocksize=gfsize
    A=[complex(zeros(gfsize,gfsize)) for i=-nmax:nmax] 
    B=[complex(zeros(gfsize,gfsize)) for i=-nmax:nmax]
    C=[complex(zeros(gfsize,gfsize)) for i=-nmax:nmax]
    # fill first block
    g_imp_max=g(E+nmax*Omega,Xvec,LolB,V)
    B[1]=-g_imp_max*Vm
    C[1]=(Id_2Nimp-g_imp_max*V0)
    for n=-nmax+1:nmax-1 #loop over block main/upper/lower diagonal
        g_imp=g(E-n*Omega,Xvec,LolB,V)
        # diagonal 
        C[n+nmax+1]=(Id_2Nimp-g_imp*V0)
        # upper minor diagonal
        B[n+nmax+1]=-g_imp*Vm
        # lower minor diagonal
        A[n+nmax+1]=-g_imp*Vp
    end
    # fill last block
    g_imp_min=g(E-nmax*Omega,Xvec,LolB,V)
    C[end]=(Id_2Nimp-g_imp_min*V0)
    A[end]=-g_imp_min*Vp
    
    return A,B,C
end
#######################

########################################################
########################################################
########################################################

########################################################
########              DOS                       ########
########################################################

# set up and solve for GF in impurity postition representation
function Gndirect(E,V,M0,Mp,Mm,Xvec,Omega,nmax,LolB=8)
    ##############################################################################################
    # E: Energy in units of Delta                                                                #
    # V: potential part of a single impurity                                                     #
    # M0,Mp,Mm: fourier components of the magnetic part of a single impurity (2x2)               #
    # Xvec: impurity distibution                                                                 #
    # Omega: Driving frequency in units of Delta/hbar                                            #
    # nmax: biggest Fourier index                                                                #
    # Lolb: length parameter                                                                     #
    ##############################################################################################
    
    ## number of impurities and GF matrix size
    Nimp=length(Xvec)
    gfsize=2*Nimp
    
    ## set up matrix for matrix eq./LHS ##
    A,B,C=LHS(E,V,M0,Mp,Mm,Xvec,Omega,nmax,Nimp,gfsize,LolB)
    #######################
    
    ## set up right side of matrix eq. ##
    b=[complex(zeros(2*Nimp,2*Nimp)) for i=-nmax:nmax] # RHS
    b[nmax+1]=g(E,Xvec,LolB,V)
    #######################
    
    ## solve matrix eq. ##
    G_N=thomas!(A,B,C,b,gfsize,2*nmax+1)
    #######################

    return G_N
end

# extract DOS from full GF
function DOSfc_from_G_N_full(G_N,cent,fc)
    -imag(tr(G_N[fc][2*cent-1:2*cent,2*cent-1:2*cent]))
end

########################################################
########################################################
########################################################

########################################################
########         Transmission/transport         ########
########################################################
# for transport calculations: regular lattice version, GF for propagation from one end of the impurity region to the other and more

function G_lattice_dos(E,V,M0,Mp,Mm,Xvec,cent,Omega,nmax,LolB=8)
    ##############################################################################################
    # INPUTS:                                                                                    #
    # E: Energy in units of Delta                                                                #
    # V: potential part of a single impurity                                                     #
    # M0,Mp,Mm: fourier components of the magnetic part of a single impurity (2x2)               #
    # Xvec: impurity distibution                                                                 #
    # Omega: Driving frequency in units of Delta/hbar                                            #
    # nmax: biggest Fourier index                                                                #
    # Lolb: length parameter                                                                     #
    ##############################################################################################
    # OUTPUTS:                                                                                   #
    # notation convention: subscribts are "to<-from" i.e G_RL is the Green's function for        #
    # particles coming from LEFT (0.0) being propagated to RIGHT (1.0)                           #
    # G_RL: Green's function from LEFT to RIGHT                                                  #
    # G_LR: Green's function from RIGHT to LEFT                                                  #
    # DOS_cimp: DOS calculated at the center impurity                                            #
    # G_LL: Green's function from LEFT to LEFT                                                   #
    # G_RR: Green's function from RIGHT to RIGHT                                                 #
    # G_cc: Green's function from CENTER to CENTER (=0.5, center of the impurity region)         #
    ############################################################################################## 
    
    ## number of impurities and GF matrix size
    Nimp=length(Xvec)
    gfsize=2*Nimp
    
    ## set up matrix for matrix eq./LHS ##
    A,B,C=LHS(E,V,M0,Mp,Mm,Xvec,Omega,nmax,Nimp,gfsize,LolB)
    #######################
    
    ## set up right side of matrix eq. and factors for final sum ##
    b=[complex(zeros(2*Nimp,2*Nimp)) for i=-nmax:nmax] # RHS
    g_RL=[[complex(zeros(2,2)) for i=1:Nimp] for k=1:2*nmax+1]# final sum factors for propagation to RIGHT
    g_LR=[[complex(zeros(2,2)) for i=1:Nimp] for k=1:2*nmax+1] # final sum factors for propagation to LEFT
    g_cc=[[complex(zeros(2,2)) for i=1:Nimp] for k=1:2*nmax+1] # final sum factors for propagation to center
    for i in 1:Nimp
        b[nmax+1][2*i-1:2*i,end-1:end]=gR_V(E,Xvec[i],0.0,LolB,V,Xvec) # RHS for G_RL: propagation from LEFT
        b[nmax+1][2*i-1:2*i,1:2]=gR_V(E,Xvec[i],1.0,LolB,V,Xvec) # RHS for G_LR: propagation from RIGHT
        b[nmax+1][2*i-1:2*i,Nimp:Nimp+1]=gR_V(E,Xvec[i],Xvec[cent],LolB,V,Xvec) # RHS for G at most centered impurity for DOS
        b[nmax+1][2*i-1:2*i,Nimp+2:Nimp+3]=gR_V(E,Xvec[i],0.5,LolB,V,Xvec) # RHS for G(0.5,0.5) for DOS
        for n=-nmax:nmax
            g_RL[n+nmax+1][i]=gR_V(E-n*Omega,1.0,Xvec[i],LolB,V,Xvec) # final sum factors for propagation to RIGHT
            g_LR[n+nmax+1][i]=gR_V(E-n*Omega,0.0,Xvec[i],LolB,V,Xvec) # final sum factors for propagation to LEFT
            g_cc[n+nmax+1][i]=gR_V(E-n*Omega,0.5,Xvec[i],LolB,V,Xvec) # final sum factors for propagation to center
        end
    end
    #######################
    
    ## solve matrix eq. ##
    G_N_full=thomas!(A,B,C,b,gfsize,2*nmax+1)
    #######################
    
    ## get rid of unnecessary components in G_N_full and introduce 2x2 blocks with imp-indices
    # indexing scheme: G_N[fourier index][imp index] (technically vector of vectors of matrices)
    G_N_RL=[[G_N_full[n][2*i-1:2*i,end-1:end] for i=1:Nimp] for n=1:2*nmax+1]
    G_N_LR=[[G_N_full[n][2*i-1:2*i,1:2] for i=1:Nimp] for n=1:2*nmax+1]
    G_N_center=[[G_N_full[n][2*i-1:2*i,Nimp+2:Nimp+3] for i=1:Nimp] for n=1:2*nmax+1]
    DOS_cimp=-imag(tr(G_N_full[nmax+1][2*cent-1:2*cent,Nimp:Nimp+1]))
    #######################
        
    ## Fourier components of the R-L-GF
    G_RL=[complex(zeros(2,2)) for i=-nmax:nmax] # from LEFT to RIGHT
    G_LR=[complex(zeros(2,2)) for i=-nmax:nmax] # from RIGHT to LEFT
    G_LL=[complex(zeros(2,2)) for i=-nmax:nmax] # from LEFT to LEFT
    G_RR=[complex(zeros(2,2)) for i=-nmax:nmax] # from RIGHT to RIGHT
    G_cc=[complex(zeros(2,2)) for i=-nmax:nmax] # from center to center

    ## insert into "end-to-end"-equation ##
    for n=-nmax+1:nmax-1
        G_RL[n+nmax+1]=sum([g_RL[n+nmax+1][i]*(M0*G_N_RL[n+nmax+1][i]+Mp*G_N_RL[n+nmax][i]+Mm*G_N_RL[n+nmax+2][i]) for i in 1:Nimp])
        G_LR[n+nmax+1]=sum([g_LR[n+nmax+1][i]*(M0*G_N_LR[n+nmax+1][i]+Mp*G_N_LR[n+nmax][i]+Mm*G_N_LR[n+nmax+2][i]) for i in 1:Nimp])
        G_LL[n+nmax+1]=sum([g_LR[n+nmax+1][i]*(M0*G_N_RL[n+nmax+1][i]+Mp*G_N_RL[n+nmax][i]+Mm*G_N_RL[n+nmax+2][i]) for i in 1:Nimp])
        G_RR[n+nmax+1]=sum([g_RL[n+nmax+1][i]*(M0*G_N_LR[n+nmax+1][i]+Mp*G_N_LR[n+nmax][i]+Mm*G_N_LR[n+nmax+2][i]) for i in 1:Nimp])
        G_cc[n+nmax+1]=sum([g_cc[n+nmax+1][i]*(M0*G_N_center[n+nmax+1][i]+Mp*G_N_center[n+nmax][i]+Mm*G_N_center[n+nmax+2][i]) for i in 1:Nimp])
    end
    # correct 0-th component 
    G_RL[nmax+1]=gR_V(E,1.0,0.0,LolB,V,Xvec)+G_RL[nmax+1]
    G_LR[nmax+1]=gR_V(E,0.0,1.0,LolB,V,Xvec)+G_LR[nmax+1]
    G_LL[nmax+1]=gR_V(E,0.0,0.0,LolB,V,Xvec)+G_LL[nmax+1]
    G_RR[nmax+1]=gR_V(E,1.0,1.0,LolB,V,Xvec)+G_RR[nmax+1]
    G_cc[nmax+1]=gR_V(E,0.5,0.5,LolB,V,Xvec)+G_cc[nmax+1]
    # insert into "end-to-end"-equation for outermost Fourier components ##
    G_RL[1]=sum([g_RL[1][i]*(M0*G_N_RL[1][i]+Mm*G_N_RL[2][i]) for i in 1:Nimp])
    G_RL[end]=sum([g_RL[end][i]*(M0*G_N_RL[end][i]+Mp*G_N_RL[end-1][i]) for i in 1:Nimp])
    G_LR[1]=sum([g_LR[1][i]*(M0*G_N_LR[1][i]+Mm*G_N_LR[2][i]) for i in 1:Nimp])
    G_LR[end]=sum([g_LR[end][i]*(M0*G_N_LR[end][i]+Mp*G_N_LR[end-1][i]) for i in 1:Nimp])
    G_LL[1]=sum([g_LR[1][i]*(M0*G_N_RL[1][i]+Mm*G_N_RL[2][i]) for i in 1:Nimp])
    G_LL[end]=sum([g_LR[end][i]*(M0*G_N_RL[end][i]+Mp*G_N_RL[end-1][i]) for i in 1:Nimp])
    G_RR[1]=sum([g_RL[1][i]*(M0*G_N_LR[1][i]+Mm*G_N_LR[2][i]) for i in 1:Nimp])
    G_RR[end]=sum([g_RL[end][i]*(M0*G_N_LR[end][i]+Mp*G_N_LR[end-1][i]) for i in 1:Nimp])
    G_cc[1]=sum([g_cc[1][i]*(M0*G_N_center[1][i]+Mm*G_N_center[2][i]) for i in 1:Nimp])
    G_cc[end]=sum([g_cc[end][i]*(M0*G_N_center[end][i]+Mp*G_N_center[end-1][i]) for i in 1:Nimp])
    #######################
    
    return G_RL,G_LR,DOS_cimp,G_LL,G_RR,G_cc 
end

### Transmission ###

function T_DOS(E,V,M0,Mp,Mm,Xvec,cent,Omega,nmax,LolB=8)
    T_lr,T_rl,DOS,T_k_diff=0.0,0.0,0.0,zeros(2*nmax+1)
    Tkl,Tkr,D,Gll,Grr,Gcc=G_lattice_dos(E,V,M0,Mp,Mm,Xvec,cent,Omega,nmax,LolB)
    for k=-nmax:nmax
        T_lr+=abs2(Tkl[k+nmax+1][1,1])
        T_rl+=abs2(Tkr[k+nmax+1][2,2])
        T_k_diff[k+nmax+1]=abs2(Tkl[k+nmax+1][1,1])-abs2(Tkr[k+nmax+1][2,2])
        if k==0
            DOS=D
        end
    end
    return T_lr,T_rl,DOS,T_k_diff
end

function Tk_DOS(E,V,M0,Mp,Mm,Xvec,cent,Omega,nmax,LolB=8)
    T_lr,T_rl,DOS,T_k_diff=zeros(2*nmax+1),zeros(2*nmax+1),0.0,zeros(2*nmax+1)
    Tkl,Tkr,D,Gll,Grr,Gcc=G_lattice_dos(E,V,M0,Mp,Mm,Xvec,cent,Omega,nmax,LolB)
    for k=-nmax:nmax
        T_lr[k+nmax+1]=abs2(Tkl[k+nmax+1][1,1])
        T_rl[k+nmax+1]=abs2(Tkr[k+nmax+1][2,2])
        T_k_diff[k+nmax+1]=abs2(Tkl[k+nmax+1][1,1])-abs2(Tkr[k+nmax+1][2,2])
        if k==0
            DOS=D
        end
    end
    return T_lr,T_rl,DOS,T_k_diff
end
########################################################
########################################################
########################################################

# functions to write the output of G_lattice_dos into a dataframe (unfolds all the matrices, etc.) subscribt indices namde to<-from
function write_to_df_full(G_Erange,Erange,nmax)
    A=[] # initialize vector for columns of the dataframe
    A_names=[] # initialize vector for names of columns of the dataframe
    dir=["RL" "LR" "" "LL" "RR" "CC"] # for names
    ind=["uu" "du" "ud" "dd"] # for names
    push!(A,Erange) # first column is energy
    push!(A_names,"E")
    push!(A,[G_Erange[iE][3] for iE=eachindex(Erange)]) # second column is DOS at center impurity
    push!(A_names,"DOS_centerimp")
    for k in [1 2 4 5 6]
        for s in 1:4
            for n in 1:2*nmax+1
                push!(A,[G_Erange[iE][k][n][s] for iE=eachindex(Erange)])
                push!(A_names,"G"*string(n-nmax-1)*dir[k]*ind[s])
            end
        end
    end
    dfA=DataFrame(A)
    names!(dfA,Symbol.(A_names))
    dfA.T_RL=sum([abs2.(dfA[j]) for j in ["G"*string(n)*"RLuu" for n=-nmax:nmax]])
    dfA.T_LR=sum([abs2.(dfA[j]) for j in ["G"*string(n)*"LRdd" for n=-nmax:nmax]])
    dfA.DOS=-imag.(dfA["G0CCuu"]+dfA["G0CCdd"])
    return dfA
end

# write this dataframe into a CSV file using: CSV.write(outputfile,df)

# read from the csv with proper types for the columns
function read_from_csv_full(file_name,nmax)
    return CSV.read(file_name,DataFrame,types=[Float64, Float64, [Complex{Float64} for i=1:4*(2*nmax+1)*5]...,Float64, Float64, Float64])
end

function write_to_df(G_Erange,Erange,nmax)
    A=[] # initialize vector for columns of the dataframe
    A_names=[] # initialize vector for names of columns of the dataframe
    dir=["RL" "LR" "" "LL" "RR" "CC"] # for names
    ind=["uu" "du" "ud" "dd"] # for names
    push!(A,Erange) # first column is energy
    push!(A_names,"E")
    push!(A,[G_Erange[iE][3] for iE=eachindex(Erange)]) # second column is DOS at center impurity
    push!(A_names,"DOS_centerimp")
    for indices in [(1,1), (2,4), (4,2), (5,3), (6,1), (6,4) ]
        k,s=indices
        for n in 1:2*nmax+1
            push!(A,[G_Erange[iE][k][n][s] for iE=eachindex(Erange)])
            push!(A_names,"G"*string(n-nmax-1)*dir[k]*ind[s])
        end
    end
    dfA=DataFrame(A)
    names!(dfA,Symbol.(A_names))
    dfA.T_RL=sum([abs2.(dfA[j]) for j in ["G"*string(n)*"RLuu" for n=-nmax:nmax]])
    dfA.T_LR=sum([abs2.(dfA[j]) for j in ["G"*string(n)*"LRdd" for n=-nmax:nmax]])
    dfA.DOS=-imag.(dfA["G0CCuu"]+dfA["G0CCdd"])
    return dfA
end

# write this dataframe into a CSV file using: CSV.write(outputfile,df)

# read from the csv with proper types for the columns
function read_from_csv(file_name,nmax)
    return CSV.read(file_name,DataFrame,types=[Float64, Float64, [Complex{Float64} for i=1:(2*nmax+1)*6]...,Float64, Float64, Float64])
end

end #module