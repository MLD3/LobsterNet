export CUDA_VISIBLE_DEVICES=""
export TS_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
declare -a model_arr=("sbd-tlearner" "sbd-dragon" "cfd-tlearner" "lobster")

pueue parallel 20
declare -a data_args_arr=("--non_compliance_type one-sided --prescriptions NIT")
declare -a nc_rate_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for rep in $(seq 0 19); do 
    for data_args in "${data_args_arr[@]}"; do
        for rate in "${nc_rate_arr[@]}"; do
            for model_args in "${model_args_arr[@]}"; do
                args="--rep $rep --model $model_args --non_compliance_rate $rate $data_args"
                pueue add taskset -c 0-50 python main_amruti.py $args
            done
        done
    done
done

# run two-sided non-compliance experiments
declare -a data_args_arr=("--non_compliance_type two-sided --prescriptions NIT CIP")
declare -a nc_rate_arr=(0.05 0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.65 0.7 0.8 0.9 0.95)
for rep in $(seq 0 19); do 
    for data_args in "${data_args_arr[@]}"; do
        for rate in "${nc_rate_arr[@]}"; do
            for model_args in "${model_args_arr[@]}"; do
                args="--rep $rep --model $model_args --non_compliance_rate $rate $data_args"
                pueue add taskset -c 0-50 python main_amruti.py $args
            done
        done
    done
done