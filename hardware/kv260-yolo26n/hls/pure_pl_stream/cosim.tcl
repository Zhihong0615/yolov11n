set origin [file dirname [file normalize [info script]]]
open_project $origin/build
set_top conv3x3_int8_stream
add_files -tb $origin/test_conv3x3_int8_stream.cpp -cflags "-std=c++14"
open_solution solution1
cosim_design -rtl verilog -tool xsim -trace_level none
exit
