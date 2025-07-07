sshpass -p 'khadas' scp run_py.cpp khadas@192.168.2.2:/home/khadas/torch/
sshpass -p 'khadas' scp CMakeLists.txt khadas@192.168.2.2:/home/khadas/torch/
sshpass -p 'khadas' scp -r cmake-android-out khadas@192.168.2.2:/home/khadas/torch/
sshpass -p 'khadas' scp bert-base-uncased.pte khadas@192.168.2.2:/home/khadas/torch/models/