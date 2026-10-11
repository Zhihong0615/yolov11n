set origin [file dirname [file normalize [info script]]]
open_project $origin/build_tree8
add_files -tb $origin/test_attention_online.cpp -cflags "-std=c++14 -I$origin -DTEST_TREE8"
open_solution solution1
csim_design -O
exit
