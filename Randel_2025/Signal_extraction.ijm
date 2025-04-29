//run("Brightness/Contrast...");
run("Enhance Contrast", "saturated=0.35");

run("Subtract...", "value=100 stack");
//run("Threshold...");
setAutoThreshold("Default dark");
setOption("BlackBackground", false);
run("Convert to Mask", "background=Dark black create");


run("Measure Stack...");