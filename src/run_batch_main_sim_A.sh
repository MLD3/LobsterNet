
export CUDA_VISIBLE_DEVICES=""
export TS_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
n_parallel=100
n=1000
p=30
u_ay=0
u_ty=0
amp=10
pueue parallel $n_parallel

declare -a model_arr=("sbd-tlearner" "cfd-tlearner")
# run one-sided non-compliance
declare -a nc_type_arr=("one-sided")
declare -a nc_rate_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for nc_type in ${nc_type_arr[@]}; do
   for nc_rate in ${nc_rate_arr[@]}; do
      for model in "${model_arr[@]}"; do
         for rep in {0..19}; do
            args="--n $n --p $p --non_compliance_type $nc_type --non_compliance_rate $nc_rate --rep 
                  $rep --model $model --u_ay $u_ay --u_ty $u_ty --amplitude $amp"
            pueue add taskset -c 0-$n_parallel python main_sim_A.py $args
         done
      done
   done
done

# run one-sided non-compliance
declare -a nc_type_arr=("two-sided")
declare -a nc_rate_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
for nc_type in ${nc_type_arr[@]}; do
   for nc_rate in ${nc_rate_arr[@]}; do
      for model in "${model_arr[@]}"; do
         for rep in {0..19}; do
            args="--n $n --p $p --non_compliance_type $nc_type --non_compliance_rate $nc_rate --rep 
                  $rep --model $model --u_ay $u_ay --u_ty $u_ty --amplitude $amp"
            pueue add taskset -c 0-$n_parallel python main_sim_A.py $args
         done
      done
   done
done




