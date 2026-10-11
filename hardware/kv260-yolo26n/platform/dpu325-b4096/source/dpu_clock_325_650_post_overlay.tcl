# Vitis 2024.2 compiler.userPostSysLinkOverlayTcl hook for an isolated
# DPU-only 325/650 MHz experiment. Never source this in a deployed design.
# It assumes the exact generated BD observed in experiments/dpu_clock_325_probe.

proc kv260_dpu_clock_325_650_post_overlay {} {
    set wiz [get_bd_cells -quiet /clk_wiz]
    set dpu [get_bd_cells -quiet /DPUCZDX8G_1]
    if {[llength $wiz] != 1 || [llength $dpu] != 1} {
        error "DPU clock hook: expected one /clk_wiz and one /DPUCZDX8G_1"
    }

    set out1 [get_bd_pins /clk_wiz/clk_out1]
    set out2 [get_bd_pins /clk_wiz/clk_out2]
    set slow [get_bd_nets -of_objects $out1]
    set fast [get_bd_nets -of_objects $out2]
    if {[llength $slow] != 1 || [llength $fast] != 1 || $slow eq $fast} {
        error "DPU clock hook: expected two independent generated output nets"
    }

    # Fail closed if Vitis changed the generated topology or this hook ran twice.
    foreach {pin expected_net} [list \
        /DPUCZDX8G_1/aclk $slow \
        /DPUCZDX8G_1/ap_clk_2 $fast \
        /psr_clk_wiz_clk_out1_324mhz/slowest_sync_clk $slow \
        /psr_clk_wiz_clk_out2_649mhz/slowest_sync_clk $fast] {
        if {[get_bd_nets -of_objects [get_bd_pins $pin]] ne $expected_net} {
            error "DPU clock hook: unexpected clock net at $pin"
        }
    }
    if {[get_property CONFIG.OVERRIDE_MMCM $wiz] ne "true" ||
        [expr {abs(double([get_property CONFIG.PRIM_IN_FREQ $wiz]) - 599.994)}] > 0.01 ||
        [expr {abs(double([get_property CONFIG.MMCM_CLKOUT0_DIVIDE_F $wiz]) - 4.0)}] > 0.01 ||
        [get_property CONFIG.MMCM_CLKOUT1_DIVIDE $wiz] ne "2"} {
        error "DPU clock hook: unexpected MMCM configuration; inspect BD first"
    }

    # PG338: specify high frequency first and match routing. With this MMCM
    # multiplier, CLKOUT0 /2 yields 649.9935 MHz; the 1x output is /2 again.
    set_property -dict [list \
        CONFIG.MMCM_CLKOUT0_DIVIDE_F 2.000000 \
        CONFIG.MMCM_CLKOUT1_DIVIDE 4 \
        CONFIG.CLKOUT1_MATCHED_ROUTING true \
        CONFIG.CLKOUT2_MATCHED_ROUTING true] $wiz
    disconnect_bd_net $slow $out1
    disconnect_bd_net $fast $out2
    connect_bd_net -net $slow $out2
    connect_bd_net -net $fast $out1

    validate_bd_design
    save_bd_design
    puts "DPU_CLOCK_325_650_HOOK_PASS: /clk_wiz/clk_out1=650->ap_clk_2; /clk_wiz/clk_out2=325->aclk; matched routing requested"
}

kv260_dpu_clock_325_650_post_overlay
