This readme.txt is a guide for installing downloaded software on a computer with no Internet access.


Table of Contents
-----------------
1. Copy downloaded software
2. Install on Windows
3. Install on Mac or Linux

1. Copy downloaded software
---------------------------
Copy this entire folder to a shared network drive or removable media, such as a USB drive. Then access the copied folder from the computer on which you want to run the Installer.

2. Install on Windows
---------------------
To run the Installer, double-click install_supportsoftware.exe. This launches MATLAB and begins the installation process.

3. Install on Mac or Linux
--------------------------
Run the Installer in a Unix shell, and specify the path to your MATLAB installation:
    ./install_supportsoftware.sh -matlabroot [install_location]
For example:
   ./install_supportsoftware.sh -matlabroot /usr/local/MATLAB/R2016b

Note: Support Packages are specific to the version of MATLAB that they were created for. 

