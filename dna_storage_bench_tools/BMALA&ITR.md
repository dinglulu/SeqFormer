## 1、部署bmala和itr

reconstruction文件夹放在测试代码同路径下

```
cd Reconstruction/BMALA

g++ -std=c++11 BMALookahead.cpp -o BMALA

cd ../Iterative

g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o LCS2.o LCS2.cpp
  g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o EditDistance.o EditDistance.cpp
  g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o Clone.o Clone.cpp
  g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o Cluster2.o Cluster2.cpp
  g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o LongestPath.o LongestPath.cpp
  g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o CommonSubstring2.o CommonSubstring2.cpp
  g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o DividerBMA.o DividerBMA.cpp
  g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o DNA.o DNA.cpp
  g++ -o DNA *.o
```



路径：



## 2、ITR_RESULT.py和BMALA_RESULT.py



### 输入

#### bmala

分别是cluster_data,bmala_path,bmala_exe_path

cluster_data是josn格式，类似：

```
   cluster_data = {
        'cluster_id1': {
            'seqs': [
                'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAG',
                'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAG'
            ],
            'quals': [
                'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII',
                'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
            ],
            'refs': 'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAG'
        },
        'cluster_id2': {
            'seqs': [
                'GGTGAGCTGCTAATGTCCCATATCGTCC',
                'GGTGAGCTGCTAATGTCCCATATCGTCC'
            ],
            'quals': [
                'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII',
                'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
            ],
            'refs': 'GGTGAGCTGCTAATGTCCCATATCGTCC'
        }
    }
```

##### 运行实例(放在测试文件的同目录下)

```
import BMALA_RESULT
bmala_path = os.path.join("./Reconstruction", "BMALA")  # BMALA 的路径
bmala_exe_path = os.path.join("./Reconstruction", "BMALA", "BMALA")  # BMALA 可执行文件的路径
success_proportion,fail_proportion,total_distance=BMALA_RESULT.BMALA_TEST(cluster_data,bmala_path,bmala_exe_path)
```



#### itr

输入分别是cluster_data,itr_path,itr_exe_path



##### 运行实例(放在测试文件的同目录下)

```
import ITR_RESULT
itr_path = os.path.join("./Reconstruction", "Iterative")  # itr 的路径
itr_exe_path = os.path.join("./Reconstruction", "Iterative", "DNA")  # itr 可执行文件的路径
success_proportion, fail_proportion, total_distance=ITR_RESULT.ITR_TEST(cluster_data,itr_path,itr_exe_path)
```



### 输出：

success_proportion, fail_proportion, total_distance

成功完全重建的类的占比，失败的占比，失败中总的编辑距离

成功类占比: 50.00%
失败类占比: 50.00%
总编辑距离: 2

