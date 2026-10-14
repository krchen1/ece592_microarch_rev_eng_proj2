# ece592_microarch_rev_eng_proj2
Challenges of Reverse Engineering Cache, Future of Cache Organization, and End of Predicted Eras

# Identification Commands
```hostname``` \
```uname -a``` \
```grep -m1 -E 'model name|Hardware|Processor' /proc/cpuinfo``` \
```lscpu -e=CPU,CORE,SOCKET,NODE``` \
```top``` \
```htop```

# Build (flush_flush example)
```cd flush_flush``` \
```gcc -O0 -o ff flush_flush.c -lm```                  (x86) \
```gcc -O0 -march=armv8-a -o ff flush_flush.c -lm```   (ARM) \

# Calibrate (frozen threshold + polarity printed to stderr):
```taskset -c <core> ./ff calibrate > <machine>_calib.csv 2> <machine>_calib_summary.txt``` \

# Measure Attack
```taskset -c <core> ./ff measure <threshold> <below|above> > <machine>_measure_data.csv 2> <machine>_measure_summary.txt``` \

# DMP Check
```cd dmp``` \
```run dmp.sh``` \
