!/bin/bash

# Install Ubuntu drivers common package
apt install ubuntu-drivers-common -y

recommended_driver=$(ubuntu-drivers devices | grep 'nvidia' | cut -d ',' -f 1 | grep 'recommended')
package_name=$(echo $recommended_driver | awk '{print $3}')
apt install $package_name -y

# Install GCC compiler for CUDA install
apt install gcc -y

# Get the release version of Ubuntu
RELEASE_VERSION=$(lsb_release -rs | sed 's/\([0-9]\+\)\.\([0-9]\+\)/\1\2/')

# Download and install CUDA package for Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${RELEASE_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

# Update and upgrade the system again to ensure all packages are installed correctly
apt update
apt install cuda -y
apt install nvidia-cuda-toolkit -y

# Add PATH and LD_LIBRARY_PATH environment variables for CUDA in .bashrc file
echo 'export PATH="/usr/bin:/bin:$PATH/usr/local/cuda/bin\${PATH:+:\${PATH}}"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

#Installing Docker binding for Nvidia

if command -v docker &> /dev/null; then
  echo "Docker is installed."
  apt install -y nvidia-docker2
  systemctl restart docker
else
  echo "Docker is not installed."
fi

#Reboot the system for enable kernel modules
reboot
