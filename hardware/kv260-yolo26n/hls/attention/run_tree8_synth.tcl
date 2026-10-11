set origin [file dirname [file normalize [info script]]]
open_project -reset $origin/build_tree8
set_top psa_attention_cached_tree8
add_files $origin/attention_cached_tree8.cpp -cflags "-std=c++14 -I$origin"
open_solution -reset solution1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period 5 -name default
csynth_design
exit
