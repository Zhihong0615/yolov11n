# Preflight for a Vitis --from_step vpl.impl resume. That resume skips the
# config_hw_runs step which normally sources _vivado_impl_props.tcl, so the
# implementation strategy must also be saved in the existing Vivado project.
if {[llength $argv] != 1} {
    error "usage: vivado -mode batch -source $argv0 -tclargs <project.xpr>"
}
set project_file [lindex $argv 0]
set project_file [file normalize $project_file]
open_project $project_file
set run [get_runs impl_1]
if {[llength $run] != 1} {
    error "expected exactly one impl_1 run in $project_file"
}
# VPL's generated script sets the run strategy, then restores its pre/post
# implementation hooks. Setting only strategy silently clears those hooks.
set vpl_dir [file dirname [file dirname $project_file]]
set previous_dir [pwd]
cd $vpl_dir
set_property strategy Performance_NetDelay_low $run
foreach {property hook_name} {
    STEPS.INIT_DESIGN.TCL.PRE _full_init_pre.tcl
    STEPS.INIT_DESIGN.TCL.POST _full_init_post.tcl
    STEPS.OPT_DESIGN.TCL.PRE _full_opt_pre.tcl
    STEPS.OPT_DESIGN.TCL.POST _full_opt_post.tcl
    STEPS.PLACE_DESIGN.TCL.PRE _full_place_pre.tcl
    STEPS.PLACE_DESIGN.TCL.POST _full_place_post.tcl
    STEPS.ROUTE_DESIGN.TCL.POST _full_route_post.tcl
    STEPS.POST_ROUTE_PHYS_OPT_DESIGN.TCL.POST _full_post_route_phys_opt_post.tcl
    STEPS.WRITE_BITSTREAM.TCL.PRE _full_write_bitstream_pre.tcl
    STEPS.WRITE_BITSTREAM.TCL.POST _full_write_bitstream_post.tcl
} {
    set file [get_files -of_object [get_filesets utils_1] "scripts/impl_1/$hook_name"]
    if {[llength $file] != 1} {
        error "expected existing Vitis hook file scripts/impl_1/$hook_name"
    }
    set_property -name $property -value $file -objects $run
}
cd $previous_dir
set observed [get_property strategy $run]
if {$observed ne "Performance_NetDelay_low"} {
    error "implementation strategy did not take effect: $observed"
}
foreach property {
    STEPS.INIT_DESIGN.TCL.PRE STEPS.INIT_DESIGN.TCL.POST
    STEPS.OPT_DESIGN.TCL.PRE STEPS.OPT_DESIGN.TCL.POST
    STEPS.PLACE_DESIGN.TCL.PRE STEPS.PLACE_DESIGN.TCL.POST
    STEPS.ROUTE_DESIGN.TCL.POST
} {
    set hook [get_property $property $run]
    if {$hook eq ""} {
        error "lost Vitis implementation hook $property"
    }
    puts "VERIFIED_IMPL_HOOK=$property:$hook"
}
# Vivado persists a project run property in the .xpr as set_property executes.
# save_project_as would create a second project and is intentionally avoided.
puts "VERIFIED_IMPL_STRATEGY=$observed"
puts "PROJECT=$project_file"
close_project
