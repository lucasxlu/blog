---
title: "[Algorithm] Sort"
date: 2018-08-02 11:37:29
mathjax: true
tags:
- Algorithm
- Data Structure
- Graph
catagories:
- Algorithm
- Data Structure
- Graph
---
## 初级排序算法
### 选择排序
首先找到数组中最小的那个元素，其次，将它和数组中的第一个元素交换位置（如果第一个元素就是最小元素就和自己交换）。再次，在剩下的元素中找到最小的元素，将它与数组中第二个元素交换位置。如此往复，直至将整个数组排序。这叫做 __选择排序__，因为它总是在不断选择剩余元素中最小的元素。

> 对于长度为$N$的数组，选择排序需要大约$N^2/2$次比较与$N$次交换。

```java
public int[] selectSort (int[] input) {
    for (int i = 0; i < input.length; i++) {
        int min = i;
        for (int j = i + 1; j < input.length; j++) {
            if (input[j] < input[min]) {
                min = j;
            }
        }
        if (input[min] != input[i]) {
            int tmp = input[min];
            input[min] = input[i];
            input[i] = tmp;
        }
    }

    return input;
}
```
* 选择排序的运行时间和输入无关，为了找到最小元素而扫描一遍数组并不能为下一遍扫描提供什么信息。
* 选择排序的数据移动是最少的，每次交换都会改变两个数组元素的值，因此选择排序用了$N$次交换——交换次数和数组大小是线性关系。

### 插入排序
为了给要插入的元素腾出空间，我们需要将其余所有元素在插入之前都向右移动一位。与选择排序一样，当前索引左边的所有元素都是有序的，但它们的最终位置还不确定，为了给更小的位置腾出空间，它们可能会被移动。但是当索引到达数组最右端，数组排序就完成了。__与选择排序不同的是，插入排序所需的时间取决于输入中元素的初始顺序__。

> 对于随机排列的长度为$N$且主键不重复的数组，平均情况下插入排序需要$\sim N^2/4$次比较以及$\sim N^2/4$次交换。最坏情况下需要$\sim N^2/2$比较和$\sim N^2/2$次交换。最好情况下需要$N-1$次比较和0次交换。

```java
public int[] insertSort(int[] input) {
    for (int i = 0; i < input.length; i++) {
        for (int j = i; j > 0 && input[j] < input[j - 1]; j--) {
            int tmp = input[j];
            input[j] = input[j - 1];
            input[j - 1] = tmp;
        }
    }

    return input;
}
```
> 插入排序对部分有序的数组十分高效，也很适合小规模数组。

> 对于随机排序的无重复主键的数组，插入排序和选择排序的运行时间是$O(N^2)$级别的。

### 希尔排序
希尔排序是一种基于插入排序的快速排序算法，对于大规模乱序数组，插入排序很慢，因为它只会交换相邻元素，因此元素只能一点点从数组一端移动到另一端。因此，希尔排序为了加速简单地改进了插入排序，__交换不相邻的元素以对数组的局部进行排序，并最终用插入排序将局部有序的数组排序__。

希尔排序的思想是使数组中任意间隔为$h$的元素都是有序的，这样的数组被称为$h$有序数组。对于每一个$h$，用插入排序将$h$个子数组独立地排序，但因为子数组是相互独立的，一个更简单的方法是在$h-$子数组中将每个元素交换到比它大的元素之前去。只需要在插入排序的代码中将移动元素的距离由1改为$h$即可。这样，希尔排序的实现就转换为一个类似于插入排序但使用不同增量的过程。

希尔排序更高效的原因是它权衡了子数组的规模和有序性，排序之初，各个子数组都很短，排序之后子数组都是部分有序的，这两种情况都很适合插入排序。

```java
public int[] shellSort(int[] input) {
    int h = 1;
    while (h < input.length / 3)
        h = 3 * h + 1;

    while (h >= 1) {
        for (int i = h; i < input.length; i++) {
            for (int j = i; j >= h && input[j] < input[j - h]; j -= h) {
                int tmp = input[j];
                input[j] = input[j - h];
                input[j - h] = tmp;
            }

            h /= 3;
        }
    }

    return input;
}
```

> 最坏情况下，希尔排序的比较次数和$N^{3/2}$成正比。

### 归并排序
要将一个数组排序，可以先递归地将它分成两半分别排序，然后将结果归并起来。归并排序能够保证任意长度为$N$的数组排序所需的时间$NlogN$成正比；它的主要缺点是它所需要的额外空间和$N$成正比。

* 原地归并
```java
private static void merge(Comparable[] input, int low, int mid, int high) {
    int i = low, j = mid + 1;
    for (int k = low; k <= high; k++) {  // copy input[low...high] to aux[low...high]
        aux[k] = input[k];
    }

    for (int k = low; k <= high; k++) {  // merge back to input[low...high]
        if (i > mid) input[k] = aux[j++];
        else if (j > high) input[k] = aux[i++];
        else if (aux[j].compareTo(aux[i]) < 0) input[k] = aux[j++];
        else input[k] = aux[i++];
    }
}
```

* 自顶向下归并
```java
public class Merge {
    private static Comparable[] aux;

    public static void sort(Comparable[] a) {
        aux = new Comparable[a.length];
        sort(a, 0, a.length - 1);
    }

    private static void merge(Comparable[] input, int low, int mid, int high) {
        int i = low, j = mid + 1;
        for (int k = low; k <= high; k++) {  // copy input[low...high] to aux[low...high]
            aux[k] = input[k];
        }

        for (int k = low; k <= high; k++) {  // merge back to input[low...high]
            if (i > mid) input[k] = aux[j++];
            else if (j > high) input[k] = aux[i++];
            else if (aux[j].compareTo(aux[i]) < 0) input[k] = aux[j++];
            else input[k] = aux[i++];
        }
    }

    private static void sort(Comparable[] a, int lo, int hi) {
        // sort array a[lo...hi]
        if (hi <= lo) return;
        int mid = lo + (hi - lo) / 2;
        sort(a, lo, mid);
        sort(a, mid + 1, hi);
        merge(a, lo, mid, hi);
    }
}
```

> 对于长度为$N$的任意数组，自顶向下的归并排序需要$\frac{1}{2}NlgN$至$NlgN$。

* 自底向上归并  
先归并那些微型数组，然后再成对归并得到的子数组，直到我们将整个数组归并在一起。
```java
public class MergeBU {
    private static Comparable[] aux;

    public static void sort(Comparable[] a) {
        aux = new Comparable[a.length];
        for (int sz = 1; sz < a.length; sz=sz+sz) {
            for (int lo = 0; lo < a.length - sz; lo += sz + sz) {
                merge(a, lo, lo + sz - 1, Math.min(lo + 2 * sz - 1, a.length - 1));
            }

        }
    }
}
```

### 快速排序
快排的特点：它是原地排序；将长度为$N$的数组排序所需时间和$NlgN$成正比。其缺点是非常脆弱，其最坏情况下的性能只有平方级别。

快排是一种分治算法。它将一个数组分成两个子数组，将两部分独立地排序，快排和归并是互补的：归并排序将将数组分成两个子数组分别排序，并将有序的子数组归并以将整个数组排序；而快排将数组排序的方式则是当两个子数组都有序时整个数组也就自然有序了。

一般策略是：先随意地取 a[lo] 作为切分元素，即那个将会被排定的元素，然后我们从数组的左端开始向右扫描，直到找到一个大于等于它的元素，然后再从数组的右端向左开始扫描，直到找到一个小于等于它的元素。交换它们的位置。如此继续，我们就可以保证左指针 i 的左侧元素都不大于切分元素，右指针 j 的右侧元素都不小于切分元素。当两个指针相遇时，我们只需要将切分元素 a[lo]和左子数组最右侧的元素（a[j]）交换然后返回 j 即可。

> 将长度为$N$的无重复数组排序，快排平均需要$\sim 2NlgN$次比较。最多需要$\frac{N^2}{2}$次比较，但随机打乱能预防这种情况。

