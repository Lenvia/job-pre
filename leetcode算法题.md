[TOC]

## 二分查找

#### 34. 在排序数组中查找元素的第一个和最后一个位置

https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/

实现lower_bound和upper_bound

记得判断边界条件！

以lower_bound为例：

- nums[l] < x ≤ nums[r]
- l + 1 = r

首先要判断

```
if(nums[l] >= target)
	return l;
if(nums[r] < target)
	return r;
```

然后再进入while

完整代码：

```
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.empty()){
            return vector<int>{-1, -1};
        }

        int low = lower_bound(nums, target);
        int up = upper_bound(nums, target);
        fprintf(stderr,"%d, %d\n", low, up);
        // printf("%d, %d\n", low, up);
        if(nums[low]!= target || nums[up]!=target){
            return vector<int>{-1, -1};
        }
        else return vector<int>{low, up};
    }

    // 找第一次
    int lower_bound(const vector<int> &nums, int target){
        int l = 0; int r = nums.size()-1;
        int mid;
        if(nums[l] >= target)
            return l;
        if(nums[r] < target)
            return r;
        while(l+1!=r){
            mid = (l+r)/2;
            if(nums[mid] >= target){
                r = mid;
            }
            else{
                l = mid;
            }
        }
        return r;
    }

    // 找最后一次
    int upper_bound(const vector<int> &nums, int target){
        int l = 0; int r = nums.size()-1;
        int mid;
        if(nums[r]<= target)
            return r;
        if(nums[l] > target)
            return l;

        while(l+1!=r){
            // printf("%d, %d\n", l, r);
            mid = (l+r)/2;
            if(nums[mid] > target){
                r = mid;
            }
            else{
                l = mid;
            }
        }
        return l;
    }

};
```





#### <font color=red>81. 搜索旋转排序数组 II</font>

https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gwv147bpw7j30z004o0tp.jpg" alt="截屏2021-11-28 18.14.16" style="zoom:50%;" />

参考题解：

https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/solution/zai-javazhong-ji-bai-liao-100de-yong-hu-by-reedfan/



循环条件：

- nums[l] ≤ x < nums[r]
- l + 1 = r

在执行之前，需要进行若干边界判断，即数组只有0、1、2个元素的时候。

在循环中，计算mid，把整个数组分为两个部分。有一个部分一定是有序的。

可以通过 nums[l]<=nums[mid] 来判断左边是否有序，但有个特殊情况需要处理：

例如，[1, 0, 1, 1, 1]

第一次 nums[l] == nums[mid]，导致左边并不是有序的，但上面判断却成立。这时候应该直接  l++，因为有与 nums[l]相等的，所以直接把nums[l]扔掉就行了，不会造成遗漏。

当左边有序时，判断target在不在左边区间，即左边最小值是否<=target

- 如果 nums[l] > target  ，左边最小的都比它大，跳到右半边 即 l = mid
- 否则就在左边二分搜索
  - if(nums[mid]<=target) l = mid
  - else r = mid;

右边同理。



完整代码：

```
class Solution {
public:
    bool search(vector<int>& nums, int target) {
        if(nums.empty())
            return false;
        int first = nums[0];
        if(nums.size()==1){
            if(first == target){
                return true;
            }
            else return false;
        }
        if(nums.size()==2){
            if(nums[0] != target && nums[1] != target)
                return false;
            else return true;
        }

        int l = 0, r = nums.size()-1;
        int mid;
        while(l+1!=r){
            // printf("%d %d\n", l, r);
            mid = (l+r)/2;

            if(nums[l] == nums[mid]){  // 如果相等，有相同的元素，干脆抛弃l
                l++;
                continue;
            }
            
            if(nums[l]<=nums[mid]){  // 左半有序
                if(nums[l] > target){  // 左边最小的都比它大，跳到右半边
                    l = mid;
                }
                else{  // 在左边二分查找
                    if(nums[mid]<=target){
                        l = mid;
                    }
                    else{
                        r = mid;
                    }
                }
                
            }
            else{  // 右半有序
                if(nums[r]<target){  // 右边最大的都不够，搜索左边
                    r = mid;
                }
                else{  // 在右边二分搜索
                    if(nums[mid]<=target){
                        l = mid;
                    }
                    else{
                        r = mid;
                    }
                }
            }
        }
        // printf("%d %d\n", l, r);
        if(nums[l]!= target && nums[r]!=target){
            return false;
        }
        else return true;
    }
};
```





## 排序

#### <font color=purple>215. 数组中的第K个最大元素</font>

https://leetcode-cn.com/problems/kth-largest-element-in-an-array/

快速排序的优化。

（针对本题，代码还能进一步优化：根据 partition的返回值，只sort一边）

完整代码：

```
class Solution {
public:
    int partition(vector<int>& nums, int left, int right){
        // 随机选取一个主元，交换到最后
        int r = rand()%(right - left + 1) + left;  
        swap(nums[r], nums[right]);

        int i = left;
        int pivot = nums[right];  // 用最后一个作为主元
        for(int j=left; j<right; j++){
            if(nums[j]<pivot){
                int temp = nums[j];
                nums[j] = nums[i];
                nums[i] = temp;
                i++;  // i始终指向第一个不小于pivot的元素
            }
        }
        nums[right]  = nums[i];  // 将主元放到这个位置
        nums[i] = pivot;
        return i;
    }
    void quickSort(vector<int>& nums, int left, int right){
        if(left >= right)  // 这里要是大于等于！！！！
            return;
        if(left+1==right){  // 仅两个元素
            if(nums[left]>nums[right]){
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;
            }
            return;
        }

        int index = partition(nums, left, right);
        quickSort(nums, left, index-1);
        quickSort(nums, index+1, right);
        return;
    }

    int findKthLargest(vector<int>& nums, int k) {
        quickSort(nums, 0, nums.size()-1);
        return nums[nums.size()-k];
    }
};
```



#### 347. 前 K 个高频元素

https://leetcode-cn.com/problems/top-k-frequent-elements/

很简单，主要是熟悉数据结构。

我用的unordered_map，再反转make_pair输入到vector里排序。

（此外，如果换成优先队列，优先队列对于 pair<int, int> 也能自动根据第一个int大根堆排序



```
class Solution {
public:
    unordered_map<int, int>count;
    vector<pair<int, int>>count_re;
    vector<int> res;

    static bool cmp(pair<int, int>a, pair<int, int>b){
        return a.first > b.first;
    }
    vector<int> topKFrequent(vector<int>& nums, int k) {
        count.clear();
        count_re.clear();
        res.clear();
        for(int i=0; i<nums.size(); i++){
            count[nums[i]]++;
        }
        unordered_map<int, int>::iterator it = count.begin();
        for(; it!=count.end(); it++){
            count_re.push_back(make_pair(it->second, it->first));
        }
        sort(count_re.begin(), count_re.end(), cmp);
        
        for(int i=1; i<=k; i++){
            res.push_back(count_re[i-1].second);
        }
        return res;

    }
};
```



## 搜索

#### <font color=dodgerblue>695. 岛屿的最大面积</font>

https://leetcode-cn.com/problems/max-area-of-island/

DFS模版题，注意这一题visited\[i]\[j]设置为true之后，搜索完不放回。

（空间上可以优化的是直接将visited数组合并到grid里，即搜索完的grid\[i]\[j]直接赋值为0）

完整代码：

```
class Solution {
public:
    int m,n;
    
    int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    bool visited[51][51];

    int dfs(vector<vector<int>>& grid, int x, int y, int curS){
        int tempS = curS;
        int newx, newy;
        for(int i=0; i<4; i++){
            newx = x+dirs[i][0];
            newy = y+dirs[i][1];
            if(newx <0 || newx>=m || newy<0 || newy>=n)
                continue;
            if(grid[newx][newy] && !visited[newx][newy]){
                visited[newx][newy] = true;  // 这里设为true之后，搜索完不要再设置false了！
                curS = max(curS, dfs(grid, newx, newy, curS+1));  // 
            }
        }
        return curS;
    }
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        m = grid.size();
        n = grid[0].size();
        memset(visited, false, sizeof(visited));

        int maxS = 0;
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(grid[i][j] && !visited[i][j]){
                    visited[i][j] = true;
                    maxS = max(maxS, dfs(grid, i, j, 1));  // 第一个位置curS=1不是0
                }
            }
        }
        return maxS;
    }
};
```



#### <font color=orange>417. 太平洋大西洋水流问题</font>

https://leetcode-cn.com/problems/pacific-atlantic-water-flow/

思路很好的一道题，一个是逆流，一个是四边搜索。

> 这道题是要寻找一个坐标既能够到达太平洋也能到达大西洋，但是这个过程一般不是一次深度搜索就能够完成的，所以我们从各边界开始逆流进行搜索。然后用两个二维数组进行记录，相当于进行了 4 次深度搜索

完整代码：

```
class Solution {
public:
    int m, n;
    int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    vector<vector<int>>results;

    void dfs(vector<vector<int>>& heights, vector<vector<bool>>& okP, vector<vector<bool>>& okA, int x, int y, int flag){
        if(!flag)
            okP[x][y] = true;
        else okA[x][y] = true;

        int newx, newy;
        for(int k=0; k<4; k++){
            newx = x + dirs[k][0];
            newy = y + dirs[k][1];
            if(newx <0 || newx >=m || newy<0 || newy>=n)  // 越界情况
                continue;
            if(heights[x][y]<=heights[newx][newy]){  // 逆流
                if(flag == 0 && !okP[newx][newy]){  // 逆流太平洋
                    okP[newx][newy] = true;
                    dfs(heights, okP, okA, newx, newy, 0);
                }
                else if(flag == 1 && !okA[newx][newy]){  // 逆流大西洋
                    okA[newx][newy] = true;
                    dfs(heights, okP, okA, newx, newy, 1);
                }
            }
        }
    }

    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        results.clear();
        if(heights.empty()){
            return results;
        }
        m = heights.size();
        n = heights[0].size();


        vector<vector<bool>> okP(m,vector<bool>(n,false));
        vector<vector<bool>> okA(m,vector<bool>(n,false));

        // 从左右搜索
        for(int i=0; i<m; i++){
            if(!okP[i][0]){  // 左搜索
                dfs(heights, okP, okA, i, 0, 0);
            }
            if(!okA[i][n-1]){  // 
                dfs(heights, okP, okA, i, n-1, 1);
            }
        }

        // 从上下搜索
        for(int j=0; j<n; j++){
            if(!okP[0][j]){  // 上搜索
                dfs(heights, okP, okA, 0, j, 0);
            }
            if(!okA[m-1][j]){  // 下搜索
                dfs(heights, okP, okA, m-1, j, 1);
            }
        }

        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(okP[i][j] && okA[i][j]){
                    results.push_back({i, j});
                }
            }
        }

        return results;
    }
};
```



#### <font color=dodgerblue>77. 组合</font>

https://leetcode-cn.com/problems/combinations/

回溯法剪枝模版题。

一个是[v1, v2]和[v2, v1]的效果是一样的，所以每次dfs的下一位候选项要比当前位的值大，即限定了for循环的下界。

此外，要凑够k个数，当前的下标是curI，一共有n个数。

那么当 n-i+1 < k-curI时，就算把后面的数都选上，也凑不够k个。所以规定了for循环的上界。

完整代码：

```
class Solution {
public:
    unordered_map<int, bool>visited;
    vector<vector<int>> results;
    vector<int>temp;
    void dfs(int n, int k, int curV, int curI){
        if(curI == k){
            results.push_back(temp);
            return;
        }
        int upbound = n+1-k+curI;  // 剪枝，如果 n-i+1 < k-curI，那么根本凑不够k个
        for(int i=curV+1; i<=upbound; i++){
            if(!visited[i]){
                visited[i] = true;
                temp.push_back(i);
                dfs(n, k, i, curI+1);
                visited[i] = false;
                temp.pop_back();
            }
        }

        // for(int i=curV+1; i<=n; i++){
        //     // 剪枝，如果 n-i+1 < k-curI，那么根本凑不够k个
        //     if(!visited[i] && n-i+1 >= k-curI){
        //         visited[i] = true;
        //         temp.push_back(i);
        //         dfs(n, k, i, curI+1);
        //         visited[i] = false;
        //         temp.pop_back();
        //     }
        // }
    }
    vector<vector<int>> combine(int n, int k) {
        results.clear();
        if(k==0 || n<1){
            return results;
        }

        visited.clear();
        temp.clear();

        for(int i=1; i<=n; i++){
            visited[i] = true;
            temp.push_back(i);
            dfs(n, k, i, 1);
            visited[i] = false;
            temp.pop_back();
        }

        return results;

    }
};
```



#### 934. 最短的桥

https://leetcode-cn.com/problems/shortest-bridge/

思路1:

1. 可以先找到其中一片岛屿，运用DFS把它标识为2，与另一片岛屿进行区分，也防止重复遍历
2. 在1的标识过程中，把第一块岛屿的所有节点压入bfs队列
3. BFS搜索队列，逐层往外“填海造陆”，直到遇到第二片岛屿

完整代码：

```
class Solution {
public:
    struct node{
        int x;  // 横坐标
        int y;  // 纵坐标
        int step;  // 步数
        node(int a, int b, int s): x(a), y(b), step(s){}
    };

    int m,n;
    int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    queue<node> q;

    void dfs(vector<vector<int>>& grid, vector<vector<bool>>& visited, int x, int y){
        int newx, newy;
        for(int i=0; i<4; i++){
            newx = x+dirs[i][0];
            newy = y+dirs[i][1];
            if(newx <0 || newx>=m || newy<0 || newy>=n)
                continue;
            if(!visited[newx][newy] && grid[newx][newy]==1){
                visited[newx][newy] = true;
                grid[newx][newy] = 2;
                q.push(node(newx, newy, 0));  // 起始step都是0

                dfs(grid, visited, newx, newy);
            }
        }
    }

    int shortestBridge(vector<vector<int>>& grid) {
        if(grid.empty())
            return 0;

        m = grid.size();
        n = grid[0].size();
        vector<vector<bool>> visited(m, vector<bool>(n,false));
        while(!q.empty()) q.pop();

        int flag = 0;
        // 先用dfs将其中一个岛屿染成不同的颜色，并把第一个岛屿的节点全部加入bfs队列
        // visited数组也设置为true（因为bfs中不会二次访问）
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(grid[i][j]==1){  // 选一个起点进行dfs就行了，然后立刻break
                    visited[i][j] = true;
                    q.push(node(i, j, 0));  // 加入bfs队列
                    grid[i][j] = 2;  // 染成不同的颜色
                    dfs(grid, visited, i, j);

                    flag = 1;
                    break;
                }
            }
            if(flag) break;
        }
        
        int newx, newy;
        while(!q.empty()){
            node temp = q.front();
            q.pop();
            // printf("%d %d value: %d step: %d\n", temp.x, temp.y, grid[temp.x][temp.y],temp.step);
            
            if(grid[temp.x][temp.y]==1)
                return temp.step-1;
            
            for(int i=0; i<4; i++){
                newx = temp.x+dirs[i][0];
                newy = temp.y+dirs[i][1];
                if(newx <0 || newx>=m || newy<0 || newy>=n)
                    continue;
                
                if(!visited[newx][newy]){
                    visited[newx][newy] = true;
                    q.push(node(newx, newy, temp.step+1));
                }
                    
            }   
        }
        
        return 0;
    }
};
```



思路2:

DFS搜索过程中，根本不要压入第一片岛屿的节点，而是直接把周围的海洋节点压入。

1. 可以先找到其中一片岛屿，运用DFS把它标识为2，与另一片岛屿进行区分，也防止重复遍历
2. 压入岛屿周围的海洋节点
3. BFS搜索队列，逐层往外“填海造陆”，直到遇到第二片岛屿

完整代码：

```
class Solution {
public:
    struct node{
        int x;  // 横坐标
        int y;  // 纵坐标
        int step;  // 步数
        node(int a, int b, int s): x(a), y(b), step(s){}
    };

    int m,n;
    int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    queue<node> q;

    void dfs(vector<vector<int>>& grid, vector<vector<bool>>& visited, int x, int y){
        int newx, newy;
        for(int i=0; i<4; i++){
            newx = x+dirs[i][0];
            newy = y+dirs[i][1];
            if(newx <0 || newx>=m || newy<0 || newy>=n)
                continue;
            if(!visited[newx][newy]){
                visited[newx][newy] = true;
                if(grid[newx][newy]==1){  // 只染色，和继续dfs。不压入bfs队列！
                    grid[newx][newy] = 2;
                    dfs(grid, visited, newx, newy);
                }
                else{  // grid这时为0
                    q.push(node(newx, newy, 1)); // 最近的海洋step为1
                }
            }
        }
    }

    int shortestBridge(vector<vector<int>>& grid) {
        if(grid.empty())
            return 0;

        m = grid.size();
        n = grid[0].size();
        vector<vector<bool>> visited(m, vector<bool>(n,false));
        while(!q.empty()) q.pop();

        int flag = 0;
        // 先用dfs将其中一个岛屿染成不同的颜色，把每个节点 附近的海域 压入bfs
        // visited数组也设置为true（因为bfs中不会二次访问）
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(grid[i][j]==1){  // 选一个起点进行dfs就行了，然后立刻break
                    visited[i][j] = true;
                    grid[i][j] = 2;  // 染成不同的颜色
                    dfs(grid, visited, i, j);
                    flag = 1;
                    break;
                }
            }
            if(flag) break;
        }
        
        int newx, newy;
        while(!q.empty()){
            node temp = q.front();
            q.pop();
            
            if(grid[temp.x][temp.y]==1)
                return temp.step-1;
            
            for(int i=0; i<4; i++){
                newx = temp.x+dirs[i][0];
                newy = temp.y+dirs[i][1];
                if(newx <0 || newx>=m || newy<0 || newy>=n)
                    continue;
                
                if(!visited[newx][newy]){
                    visited[newx][newy] = true;
                    q.push(node(newx, newy, temp.step+1));
                }
                    
            }   
        }
        
        return 0;
    }
};
```





#### <font color=purple>126. 单词接龙 II</font>

https://leetcode-cn.com/problems/word-ladder-ii/

允许重复入队的BFS，但有条件。就是一个单词再次入队时，它的step不能高于上次入队的step。

所以使用 temp.step+1 <= visited[newWord] 来进行检验。 （visited[str]初始化为字典的最大长度）

完整代码：

```
class Solution {
public:
    struct node{
        string str;  // 字符串
        int step;  // 变换次数
        vector<string> curL;  // 当前字符串向量
        node(string str, int step, vector<string>& l): str(str), step(step), curL(l){}
    };

    unordered_map<string, int>visited;  // 这里visited指的是str的步数
    // 检查两个字符串是否只有一个字符不同  
    bool isOneDiff(string str1, string str2){
        if(str1.length()!=str2.length())
            return false;
        int dif = 0;
        for(int i=0; i<str1.length(); i++){
            if(str1[i]!=str2[i]){
                dif++;
                if(dif>1) return false;
            }
        }
        if(!dif) return false;
        return true;
    }

    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        vector<vector<string>> results;
        if(wordList.empty())  // 字典为空
            return results;
        int len = wordList.size();
        visited.clear();

        // 检查字典里有没有endWord，没有直接返回空
        bool check = false;
        for(int i=0; i<len; i++){
            visited[wordList[i]] = len;
            if(endWord==wordList[i]){
                check = true;
            }
        }
        if(!check) return results;
        

        queue<node>q;
        vector<string> l;
        l.push_back(beginWord);
        q.push(node(beginWord, 0, l));
        int maxT = len;  // 最多不会超过len次交换
        while(!q.empty()){
            node temp = q.front();
            q.pop();
            
            if(temp.step>maxT)  // 已经不是最短转换了
                break;
            if(temp.str == endWord){  // 记录结果，就不用再bfs了
                results.push_back(temp.curL);
                maxT = temp.step;
                continue;
            }
            for(int i=0; i<len; i++){
                // 如果只有一个字符差，并且只允许再次放入时的step小于等于之前的最大step
                string newWord = wordList[i];
                if(temp.step+1 <= visited[newWord] && 
                    isOneDiff(temp.str, newWord)){
                    visited[newWord] = temp.step+1;
                    vector<string> newL = temp.curL;
                    newL.push_back(newWord);
                    q.push(node(newWord, temp.step+1, newL));
                }
            }
        }

        return results;
    }
};
```





## 动态规划

#### 64. 最小路径和

https://leetcode-cn.com/problems/minimum-path-sum/

二维DP的入门。

完整代码：

```
class Solution {
public:
    int m, n;
    int minPathSum(vector<vector<int>>& grid) {
        if(grid.empty()) return 0;

        m = grid.size();
        n = grid[0].size();

        int dp[m+1][n+1];  // dp[i][j]表示到达(i,j)时的最小和

        int newx, newy;
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                dp[i][j] = grid[i][j];  // 初始化
                if(i-1>=0 && j-1>=0){  // 上面和左边都有格子
                    dp[i][j] += min(dp[i-1][j], dp[i][j-1]);
                }
                else{
                    if(i-1>=0)
                        dp[i][j] += dp[i-1][j];
                    else if(j-1>=0)
                        dp[i][j] += dp[i][j-1];
                    else continue;  // 上面左面都没格子，跳过
                }
            }
        }
        return dp[m-1][n-1];
    }
};
```



由于只依赖于当前行和上一行，可以转成一维数组进行空间优化

```
class Solution {
public:
    int m, n;
    int minPathSum(vector<vector<int>>& grid) {
        if(grid.empty()) return 0;

        m = grid.size();
        n = grid[0].size();

        int dp[n+1];
        memset(dp, 0, sizeof(dp));
        int newx, newy;
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(i-1>=0 && j-1>=0){  // 上面和左边都有格子
                    dp[j] = min(dp[j], dp[j-1]);
                }
                else{
                    if(i-1>=0)
                        dp[j] = dp[j];
                    else if(j-1>=0)
                        dp[j] = dp[j-1];
                }
                dp[j] += grid[i][j];  // 反正都要加的，放在最后
            }
        }
        return dp[n-1];
    }
};
```





#### 542. 01 矩阵

还有动态规划的解法，但是我直觉用的BFS。

这次是把所有的0节点都放入，而不是从单个0节点开始搜索。

https://leetcode-cn.com/problems/01-matrix/

```
class Solution {
public:
    struct node{
        int x;
        int y;
        node(int x, int y): x(x), y(y){}
    };

    int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        if(mat.empty()){
            return mat;
        }
        int m,n;
        m = mat.size();
        n = mat[0].size();

        vector<vector<int>> dp(m, vector(n, 30000));  // 不可能超过2*10^4
        bool visited[m][n];
        memset(visited, false, sizeof(visited));

        queue<node>q;
        // 直接遍历，把所有的0节点压入队列。
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(mat[i][j]==0){
                    dp[i][j] = 0;
                    visited[i][j] = true;
                    q.push(node(i,j));
                }
            }
        }

        // BFS。由于0节点是对等的，所以新的节点一定，是最近的。
        int newx, newy;
        while(!q.empty()){
            node temp = q.front();
            q.pop();
            for(int k=0; k<4; k++){
                newx = temp.x+dirs[k][0];
                newy = temp.y+dirs[k][1];
                if(newx<0 || newx>=m || newy<0 || newy>=n)
                    continue;
                if(!visited[newx][newy]){
                    visited[newx][newy] = true;
                    q.push(node(newx, newy));
                    dp[newx][newy] = dp[temp.x][temp.y]+1;
                }
            }
        }
        return dp;
    }
};
```



#### <font color=orange>221. 最大正方形</font>

https://leetcode-cn.com/problems/maximal-square/

全为1的最大正方形。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gx8387n6kfj30n60i83zd.jpg" alt="截屏2021-12-10 01.19.45" style="zoom:33%;" />

思路是 dp\[i]\[j]表示以(i, j)作为正方形右下角定点，形成的最大正方形的边长

如果 matrix\[i]\[j] == 0，dp\[i]\[j]=0

如果 没有左上角，dp\[i]\[j] = 1

如果 有左上角，令el为左上角的最大边长（可能为0）。从matrix\[i][j]向上向下搜索el个单位。如果遇到0，则减少el继续搜。如果没遇到0，则边长dp\[i][j]为当前el+1。

完整代码：

```
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.empty())
            return 0;
        int m, n;
        m = matrix.size();
        n = matrix[0].size();

        int dp[m+1][n+1];  // dp[i][j]表示以(i, j)作为正方形右下角定点，形成的最大正方形的边长
        memset(dp, 0, sizeof(dp));

        int el; // dp[i-1][j-1]
        int maxEl = 0;  // 最大正方形边长
        int flag; // 检验是否能延续左上角正方形
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(matrix[i][j]=='0'){
                    continue;
                }
                dp[i][j] = 1;  // 先保证自己一块
                // 左上角没有东西，那么只有自己一块
                if(i-1<0 || j-1<0){
                    maxEl = max(maxEl, dp[i][j]);  // 这里需要判断一下！！因为这时候dp非0
                    continue;
                }
                    
                // 左上角有东西，验证本位置向左和向上是否都为1
                // 只要遇到0，就break。
                el = dp[i-1][j-1];
                while(el){  // 如果当前不行，就减少el重试
                    flag = 1;  // 每次都初始化flag
                    for(int k=1; k<=el; k++){
                        if(matrix[i-k][j]=='0' || matrix[i][j-k]=='0'){
                            flag = 0;
                            break;  // 寻求更小的el
                        }
                    }
                    if(flag){
                        dp[i][j] = el+1;  // 延续左上角
                        break;  // 打破while循环
                    }
                    el--;  // 要放在最后减！否则会影响到前面的dp[i][j] = el+1
                }
                maxEl = max(maxEl, dp[i][j]);
            }
        }
        return maxEl*maxEl;
    }
};
```



**优化！**

在上面的思路上，dp\[i-1][j]相当于检验上面的最大边长，dp\[i][j-1]相当于检验左边。
$$
dp[i][j] = \min(dp[i][j], dp[i-1][j], dp[i][j-1]) + 1
$$

完整代码：

```
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.empty())
            return 0;
        int m, n;
        m = matrix.size();
        n = matrix[0].size();

        int dp[m+1][n+1];  // dp[i][j]表示以(i, j)作为正方形右下角定点，形成的最大正方形的边长
        memset(dp, 0, sizeof(dp));

        int el; // dp[i-1][j-1]
        int maxEl = 0;  // 最大正方形边长
        int flag; // 检验是否能延续左上角正方形
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(matrix[i][j]=='0'){
                    continue;
                }
                
                // 左上角没有东西，那么只有自己一块
                if(i-1<0 || j-1<0){
                    dp[i][j] = 1;  // 自己一块
                }
                else{
                    // 左上角有东西，则左和上也都有东西。验证三个点
                    dp[i][j] = min(min(dp[i-1][j-1], dp[i-1][j]), dp[i][j-1])+1;
                }
                maxEl = max(maxEl, dp[i][j]);
            }
        }
        return maxEl*maxEl;
    }
};
```



#### 279. 完全平方数

https://leetcode-cn.com/problems/perfect-squares/

给定正整数 *n*，找到若干个完全平方数使得它们的和等于 *n*。需要让组成和的完全平方数的个数最少。

如果n本身是完全平方数，那直接是1；

如果n不是，向前退：dp[i], dp[k]+dp[i-k]

未优化代码：

```
class Solution {
public:
    int numSquares(int n) {
        int dp[n+1];  // dp[i]表示组成i的完全平方数
        
        fill(dp, dp+n+1, n);  // 最坏情况也不过是n

        dp[0] = 1;
        dp[1] = 1;

        int sq;
        for(int i=2; i<=n; i++){
            sq = int(sqrt(i));
            if(sq*sq==i){  // 自身就是完全平方数 
                dp[i] = 1;
                continue;
            }
            // 否则，开始往前搜
            for(int k=1; k<=i/2; k++){
                dp[i] = min(dp[i], dp[k]+dp[i-k]);
            }
        }
        return dp[n];
    }
};
```



既然向前退，为什么不直接退大点呢？每次退一个平方数！k的范围进一步缩小为 [1, √n]

即每次退k的平方，那样的话 dp[i] = dp[k^2] + dp[i-k^2]，并且dp[k^2]为1。

完整代码：

```
class Solution {
public:
    int numSquares(int n) {
        int dp[n+1];  // dp[i]表示组成i的完全平方数
        
        fill(dp, dp+n+1, n);  // 最坏情况也不过是n

        dp[0] = 1;
        dp[1] = 1;

        int sq;
        for(int i=2; i<=n; i++){
            sq = int(sqrt(i));
            if(sq*sq==i){  // 自身就是完全平方数 
                dp[i] = 1;
                continue;
            }
            // 否则，开始往前搜
            for(int k=1; k<=sq; k++){
                dp[i] = min(dp[i], 1+dp[i-k*k]);
            }
        }
        return dp[n];
    }
};
```





#### <font color=purple>91. 解码方法</font>

https://leetcode-cn.com/problems/decode-ways/

细节！！！

dp[i]表示到i为止，可以解码的方法总数。

在和前一位组合前，需要判断合不合法！即是否大于26，是否前面已有一个0.



i==1时另外考虑，详细看代码。这里只说i>1的思路。

如果s[i]为0（必须和前面组合）

​	如果 组合>26或前一位为0，则无法组合，return 0

​	否则组合。dp[i] = dp[i-2];

若s[i]不为0

​	判断和前一位组合合不合法。

​	如果不合法，只有一种选择不组合。dp[i] = dp[i-1]

​	如果合法，则可以组合也可以不组合。dp[i] = dp[i-2] + dp[i-1]



提供几个测试用例：

```
"110"
"230"
"10011"
"2101"
"301"
——————————————输出————————————
1
0
0
1
0
```

完整代码：

```
class Solution {
public:
    int numDecodings(string s) {
        int n = s.length();
        if(n==0)
            return 0;
        if(s[0]=='0')
            return 0;
        
        int dp[n+1];  // dp[i]表示到i为止，可以解码的方法总数

        dp[0] = 1;

        for(int i=1; i<n; i++){
            if(s[i]=='0'){  // 必须要和前面的组合
                // 如果和前面的组合不合法
                int temp = stoi(s.substr(i-1, 2));
                if(temp > 26 || s[i-1]=='0')  // 越界或前面有一个0了
                    return 0;

                if(i==1) dp[i] = 1;
                else dp[i] = dp[i-2];
                    
            }
            else{ // 该位不为0
                int temp = stoi(s.substr(i-1, 2));
                if(temp > 26 || s[i-1]=='0')  // 越界或前面有一个0了，不能组
                    dp[i] = dp[i-1];
                else{  // 可以和前面组
                    if(i==1)
                        dp[i] = 2;
                    else dp[i] = dp[i-2] + dp[i-1];
                }
            }
        }
        return dp[n-1];
    }
};
```





#### 139. 单词拆分

https://leetcode-cn.com/problems/word-break/

用unordered_map

dp[i]表示截止到i能否拆

首先直接截取s[0]~s[i]看在不在

然后再挨个判断 dp[j] 和 s[j+1]~s[i]

```
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int dictLen = wordDict.size();
        if(dictLen==0)
            return false;
        int n = s.length();
        bool dp[n+1];  // dp[i]表示截止到i能否拆
        memset(dp, false, sizeof(dp));

        unordered_map<string, bool> exist;  // 方便查找，放到map里
        for(int i=0; i<dictLen; i++)
            exist[wordDict[i]] = true;

        // 边界，首个字符
        if(exist[s.substr(0, 1)]) dp[0] = true; else dp[0] = false;

        for(int i=1; i<n; i++){
            // 先检查自己可不可以
            if(exist[s.substr(0, i+1)]){
                dp[i] = true;
                continue;
            }
            // 向前抽取子串
            for(int j=i; j>=1; j--){
                // 检查dp[j]和 [j+1~i]这一段能否找到
                if(dp[j-1] && exist[s.substr(j, i-j+1)]){  // 
                    dp[i] = true;
                    break;
                }
            }

        }
        return dp[n-1];

    }
};
```





#### 300. 最长递增子序列

https://leetcode-cn.com/problems/longest-increasing-subsequence/

最长递增子序列。

完整模版：

```
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        if(n==0 || n==1)
            return n;
        int dp[n];
        
        fill(dp, dp+n, 1);  // 初始化为1

        int maxL = 1;
        for(int i=1; i<n; i++){
            for(int k=i-1; k>=0; k--){
                if(nums[i]>nums[k])
                    dp[i] = max(dp[i], dp[k]+1);
            }
            maxL = max(maxL, dp[i]);
        }
        return maxL;
    }
};
```





#### 1143. 最长公共子序列

https://leetcode-cn.com/problems/longest-common-subsequence/

最长公共子序列。填个边界，下标从1开始。

```
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int l1,l2;
        l1 = text1.length();
        l2 = text2.length();

        int dp[l1+1][l2+1];
        memset(dp, 0, sizeof(dp));

        for(int i=1; i<=l1; i++){
            for(int j=1; j<=l2; j++){
                if(text1[i-1]==text2[j-1])
                    dp[i][j] = dp[i-1][j-1]+1;
                else{
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }

        return dp[l1][l2];
    }
};
```





#### <font color=dodgerblue>416. 分割等和子集</font>

https://leetcode-cn.com/problems/partition-equal-subset-sum/

背包元素能否组成特定值的问题。

参考HDU 1069的笔记。

递推关系:dp[j] = dp[j] || dp[j-weight[i]]

> 当dp表示最大/最小值时，dp[?] = max/min (dp[??], dp[???]) 
>
> 当dp表示方法数时，dp[?] = dp[??] + dp[???] 
>
> 当dp表示有解无解时，dp[?] = dp[??] || dp[???]



完整代码：

```
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n = nums.size();
        if(n==0)
            return false;
        int sum = 0;
        for(int i=0; i<n; i++){
            sum+= nums[i];
        }
        if(sum%2==1) return false;

        bool dp[sum+1];  // dp[i]表示价值为i能否由元素组成
        memset(dp, false, sizeof(dp));
        dp[0] = true;  // 初始化开头
        for(int i=0; i<n; i++){
            for(int j=sum; j>=0; j--){
                if(j-nums[i]>=0){  // 该元素可以取
                    dp[j] = dp[j] || dp[j-nums[i]];  // 取或不取
                }
            }
        }
        return dp[sum/2];
        
    }
};
```



优化一下j的范围。

```
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n = nums.size();
        if(n==0)
            return false;
        int sum = 0;
        for(int i=0; i<n; i++){
            sum+= nums[i];
        }
        if(sum%2==1) return false;

        bool dp[sum+1];  // dp[i]表示价值为i能否由元素组成
        memset(dp, false, sizeof(dp));
        dp[0] = true;  // 初始化开头
        for(int i=0; i<n; i++){
            for(int j=sum/2; j>=nums[i]; j--){
                dp[j] = dp[j] || dp[j-nums[i]];  // 取或不取
            }
        }
        return dp[sum/2];
        
    }
};
```



#### <font color=purple>474. 一和零</font>

https://leetcode-cn.com/problems/ones-and-zeroes/

双变量的01背包

要倒着，这里dp是二维数组，是因为有两个变量。与之前的二维数组不是同一个意思，所以要从后往前。

当 strs[k]含有x个0和y个1的时候

dp\[i][j] = max(dp\[i][j], dp\[i-x][j-y]+1)  不取和取

完整代码：

```
class Solution {
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
        int dp[m+1][n+1];  // dp[i][j] 表示要求0和1的个数为i和j时，最大的子集长度

        /**
            当 strs[k]含有x个0和y个1的时候
            dp[i][j] = max(dp[i][j], dp[i-x][j-y])  不取和取
        */

        memset(dp, 0, sizeof(dp));
        int sl = strs.size();
        vector<pair<int, int>> items;
        int first, second;
        for(int i=0; i<sl; i++){
            string s = strs[i];
            first = 0; second = 0;
            for(int j=0; j<s.length(); j++){
                if(s[j]=='0')
                    first++;
                else second++;
            }
            items.push_back(make_pair(first, second));
        }
        dp[0][0] = 0;
        for(int k=0; k<sl; k++){
            // 要倒着！这里dp是二维数组，是因为有两个变量。
            // 与之前的二维数组不是同一个意思，所以要从后往前。
            for(int i=m; i>=items[k].first; i--){
                for(int j=n; j>=items[k].second; j--){
                    dp[i][j] = max(dp[i][j], dp[i-items[k].first][j-items[k].second]+1);
                }
            }
        }

        return dp[m][n];
    }
};
```

