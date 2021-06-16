using Distributed
using DataFrames, CSV
using Dates

@everywhere include("modules/EdgeImp.jl")
@everywhere using .EdgeImpV8

# input parameters, by wraping things in an apporpriate loop we can run several parameters from the same script

#impurity system
LolB=8.0 #scaling parameter for system length (to avoid tunneling)
M=0.2 # magnetic impurity strength
Ms=0.0 # ratio of static magentic moment in y-direction
M0,Mp,Mm=Mimpsingle(M,Ms)
Nimp=round(Int,LolB/M) # resulting impurity number
V=0.0 # potential strength

# Floquet stuff
nmax=2 # biggest Fourier components (=> 2nmax+1 FCs)
Omega=25 # driving frequency

# batches and runs per batch, impurity averaging is carried out for Nruns runs and then stored in a new file, this is repeated batches times
Nruns=5

# start date to name files/directories of the current run
startdate=today()

# energyrange
Emax=35.0 
numE=701 # no. of energy points
Erange=range(-Emax+V,stop=Emax+V,length=numE)

######## .log and .dat file ##############
# create directory to store data/logs in
global i=0
#Omegaint=Int(Omega)
# make sure we write in a new directory
while isdir("data/$(startdate)_$i")==true
    global i+=1
end
out_dir="data/$(startdate)_$i"
mkdir(out_dir)
# log file with parameters for the run
logfile=out_dir*"/$(startdate).log"
open(logfile,"a") do iolog
    timestamp=now()
    write(iolog, "[$timestamp]: START V8 with parameters: V=$V , LolB=$LolB , M=$M, Ms=$Ms, Omega=$Omega , nmax=$nmax , Erange=$Erange, $Nruns imp averaging runs \n")
end
# .dat file
#outputfile_TLR="data/v8/$(startdate)_$i/$(startdate)_TLR.dat"
#outputfile_TRL="data/v8/$(startdate)_$i/$(startdate)_TRL.dat"
#outputfile_DOS="data/v8/$(startdate)_$i/$(startdate)_DOS.dat"
#outputfile_Tdiff="data/v8/$(startdate)_$i/$(startdate)_Tdiff.dat"
#while isfile(outputfile)==true # check for existing file
#    outputfile=outputfile * ".more"
#end # while


    
# Transmission calculations
for run=1:Nruns
    Xvec,cent=imps(Nimp)
    G=Ek->G_lattice_dos(Ek,V,M0,Mp,Mm,Xvec,cent,Omega,nmax,LolB)
    results=pmap(G,Erange)
    # write results to outputfile
    results_df=write_to_df(results,Erange,nmax)
    CSV.write(out_dir*"/$(startdate)_$(run).csv",results_df)

    if mod(run,10)==0
        # progress log entry
        open(logfile,"a") do iolog
            timestamp=now()
            write(iolog, "[$timestamp]: Transmission run $(run) completed \n")
        end #open
    end
end

open(logfile,"a") do iolog
    timestamp=now()
    write(iolog, "[$timestamp]: DONE")
end

#function read_input_from_k_file(A)
#    B=[]
#    b=Float64[]
#    for a in A
#        if first(a)=='['
#            push!(b,parse(Float64,a[2:end]))
#        elseif typeof(a)==Float64
#            push!(b,a)
#        elseif last(a)==']'
#            push!(b,parse(Float64,a[1:end-1]))
#            push!(B,b)
#            b=Float64[]
#        end
#    end
#    return B
#end