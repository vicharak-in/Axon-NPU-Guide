#!/system/bin/bash

echo "Enable CPU idle state1 for all cores:"
for i in $(seq 0 7); do
    echo 0 > /sys/devices/system/cpu/cpu${i}/cpuidle/state1/disable
done


echo "NPU available frequencies:"
cat /sys/class/devfreq/fdab0000.npu/available_frequencies
echo "Set NPU to min frequency:"
echo userspace > /sys/class/devfreq/fdab0000.npu/governor
echo 300000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq
cat /sys/class/devfreq/fdab0000.npu/cur_freq


echo "CPU available frequencies:"
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
cat /sys/devices/system/cpu/cpufreq/policy4/scaling_available_frequencies
cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies
echo "Set CPU to min frequency:"
echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 408000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
echo userspace > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 408000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed
cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq
echo userspace > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo 408000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed
cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq


echo "GPU available frequencies:"
# GPU drivers depends on kernel
kernel_version=$(uname -r)
kernel_major=$(echo "$kernel_version" | cut -d. -f1)
if [ "$kernel_major" -eq 5 ]; then
    cat /sys/class/devfreq/fb000000.gpu/available_frequencies
    echo "Set GPU to min frequency:"
    echo userspace > /sys/class/devfreq/fb000000.gpu/governor
    echo 300000000 > /sys/class/devfreq/fb000000.gpu/userspace/set_freq
    cat /sys/class/devfreq/fb000000.gpu/cur_freq
elif [ "$kernel_major" -ge 6 ]; then
    cat /sys/class/devfreq/fb000000.gpu-panthor/available_frequencies
    echo "Set GPU to min frequency:"
    echo userspace > /sys/class/devfreq/fb000000.gpu-panthor/governor
    echo 300000000 > /sys/class/devfreq/fb000000.gpu-panthor/userspace/set_freq
    cat /sys/class/devfreq/fb000000.gpu-panthor/cur_freq
fi    


echo "DDR available frequencies:"
cat /sys/class/devfreq/dmc/available_frequencies
echo "Set DDR to min frequency:"
echo userspace > /sys/class/devfreq/dmc/governor
echo 528000000 > /sys/class/devfreq/dmc/userspace/set_freq
cat /sys/class/devfreq/dmc/cur_freq
