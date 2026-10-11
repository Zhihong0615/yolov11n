set origin [file dirname [file normalize [info script]]]
open_project -reset $origin/build
set_top conv3x3_int8_stream
add_files $origin/conv3x3_int8_stream.cpp -cflags "-std=c++14"
open_solution -reset solution1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period 10 -name default
csynth_design
exit
