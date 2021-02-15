#! /bin/bash
job_size=10
i=0
n_procs=32

while [ $i -lt $job_size ];
do
echo $i
mpirun -np $n_procs -ppn $n_procs mpi_amide_slicing.py $i $job_size
i=$(( $i + 1 ));
# adjust job_size and n_procs depending on the memory and number of parallelizable cores available
done
