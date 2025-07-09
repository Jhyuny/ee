python ours.py --teacher_model /mnt/aix7101/jeong/tinybert/teachers/sst2 \
               --task_name SST-2 \
               --do_lower_case

python ours.py --do_eval \
               --teacher_model /mnt/aix7101/jeong/tinybert/teachers/sst2 \
               --task_name SST-2 \
               --do_lower_case