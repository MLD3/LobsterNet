export CUDA_VISIBLE_DEVICES=""
export TS_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
declare -a model_arr=("sbd-tlearner" "cfd-tlearner")
declare -a nc_type_arr=("one-sided" "two-sided")
declare -a effects=(1 2 3 4 5 6 7 8 9 10 12 20) 
n_parallel=100
n=1000
p=30
amp=10


pueue parallel $n_parallel
for nc_type in ${nc_type_arr[@]}; do
   for effect in ${effects[@]}; do
      for model in "${model_arr[@]}"; do
         for rep in {0..20}; do
            args="--n $n --p $p --non_compliance_type $nc_type --effect $effect --rep $rep --model $model --amplitude $amp"
            pueue add taskset -c 0-$n_parallel python main_sim_B.py $args
         done
      done
   done
done
