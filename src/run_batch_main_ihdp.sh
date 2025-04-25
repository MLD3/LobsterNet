export CUDA_VISIBLE_DEVICES=""
export TS_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
pueue parallel 100


declare -a model_arr=("sbd-tlearner" "sbd-dragon" "cfd-tlearner" "lobster")
# run one-sided non-compliance experiments
declare -a nc_type_arr=("one-sided")
declare -a nc_rate_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for nc_type in ${nc_type_arr[@]}; do
    for nc_rate in ${nc_rate_arr[@]}; do
        for model in ${model_arr[@]}; do
            for rep in {0..19}; do 
                args="--non_compliance_type $nc_type --non_compliance_rate $nc_rate 
                      --average_effect_size 4.0 --model $model --rep $rep --type rate"
                pueue add taskset -c 0-100 python main_ihdp.py $args 
            done
        done
    done
done

# run two-sided non-compliance experiments
declare -a nc_type_arr=("two-sided")
declare -a nc_rate_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
for nc_type in ${nc_type_arr[@]}; do
    for nc_rate in ${nc_rate_arr[@]}; do
        for model in ${model_arr[@]}; do
            for rep in {0..19}; do 
                args="--non_compliance_type $nc_type --non_compliance_rate $nc_rate 
                      --average_effect_size 4.0 --model $model --rep $rep --type rate"
                pueue add taskset -c 0-100 python main_ihdp.py $args 
            done
        done
    done
done