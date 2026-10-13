# ece592_microarch_rev_eng_proj2
Challenges of Reverse Engineering Cache, Future of Cache Organization, and End of Predicted Eras

# Identification Commands
```hostname``` \
```uname -a``` \
```grep -m1 -E 'model name|Hardware|Processor' /proc/cpuinfo``` \
```lscpu -e=CPU,CORE,SOCKET,NODE``` \
```top``` \
```htop```

# Calibrate (flush_flush example)
```gcc -O0 -o flush_flush flush_flush.c``` \
```taskset -c 0 ./flush_flush calibrate > <machine>_calib.csv ```

# Build and Run Attack
```gcc -O0 -0 <attack> <attack.c> ``` \
```tasket -c <core> ./<attack> attack``` \
```taskset -c <core> ./flush_flush measure > <machine>_calib.csv 2> <machine>_measure_summary.txt```