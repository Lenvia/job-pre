# 分类







# 系列

## 剑指Offer

### 剑指 Offer 09. 用两个栈实现队列

栈1专门用来进入，栈2专门用来弹出。当栈2没元素时，把栈1里存放的都推进来。

```
class CQueue {
public:
    stack<int>st1;
    stack<int>st2;
    CQueue() {
        while(!st1.empty()) st1.pop();
        while(!st2.empty()) st2.pop();
    }
    
    void appendTail(int value) {
        st1.push(value);
    }
    
    int deleteHead() {
        int res = -1;
        if(st2.empty()){
            while(!st1.empty()){
                st2.push(st1.top());
                st1.pop();
            }
            if(st2.empty()) return -1;
        }

        res = st2.top();
        st2.pop();
    
        return res;
    }
};
```





### 剑指 Offer 30. 包含min函数的栈

```cpp
class MinStack {
public:
    /** initialize your data structure here. */
    stack<int>st1;
    stack<int>st2;

    MinStack() {
        while(!st1.empty()) st1.pop();
        while(!st2.empty()) st2.pop();
    }
    
    void push(int x) {
        st1.push(x);
        if(st2.empty() || st2.top() > x)
            st2.push(x);
        else st2.push(st2.top());
    }
    
    void pop() {
        st1.pop();
        st2.pop();
    }
    
    int top() {
        return st1.top();
    }
    
    int min() {
        return st2.top();
    }
};
```





### 剑指 Offer 35. 复杂链表的复制

方法一：递归+map

```
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/
class Solution {
public:
    unordered_map<Node*, Node*> cache;
    Node* copyRandomList(Node* head) {
        if(head == NULL) return NULL;

        if(cache.find(head) == cache.end()){
            Node* headNew = new Node(head->val);
            cache[head] = headNew;
            headNew -> next = copyRandomList(head->next);
            headNew -> random = copyRandomList(head->random);
            return headNew;
        }
        return cache[head];
    }
};
```



方法二：冗余节点

```
class Solution {
public:
    
    Node* copyRandomList(Node* head) {
        if(head == NULL) return NULL;

        // 第一遍先完成冗余节点创建
        for(Node* node = head; node!= NULL; node = node->next->next){
            Node* nodeNew = new Node(node->val);
            nodeNew->next = node->next;
            node->next = nodeNew;
        }
        // 第二遍完善random指针，不要修改next
        for(Node* node = head; node!= NULL; node = node->next->next){
            Node* nodeNew = node->next;
            nodeNew->random = node->random? node->random->next : NULL; // 注意判空
        }

        Node* headNew = head->next;
        // 第三遍复原原指针，并完善next指针
        for(Node* node = head; node!= NULL; node = node->next){
            Node* nodeNew = node->next;
            node->next = nodeNew->next;
            nodeNew->next = node->next ? node->next->next : NULL; 
        }
        return headNew;
    }
};
```







### 剑指 Offer 53 - I. 在排序数组中查找数字 I

统计一个数字在排序数组中出现的次数。

手写二分查找（存在相等元素的第一个位置和最后一个位置）

```
class Solution {
public:

    int binarySearch(vector<int>& nums, int target, bool lower) {
        int left = 0, right = (int)nums.size() - 1;

        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
                
            } else if(nums[mid] < target) {
                left = mid + 1;
            }
            else{  // 向左继续寻找
                if(lower){
                    for(int j=mid-1; j>=0; j--){
                        if(nums[j] != nums[j+1]){
                            return j+1;
                        }
                    }
                    return 0;
                }
                else{
                    for(int j=mid+1; j<nums.size(); j++){
                        if(nums[j] != nums[j-1]){
                            return j-1;
                        }
                    }
                    return nums.size()-1;
                }
            }
        }
        return -1;
    }

    int search(vector<int>& nums, int target) {
        if(nums.size() == 0) return 0;

        int index1 = binarySearch(nums, target, true);
        if(index1 == -1) return 0;
        int index2 = binarySearch(nums, target, false);

        return index2 - index1 + 1;
    }
};
```





### 剑指 Offer 53 - II. 0～n-1中缺失的数字

如果不相等，就调换位置。第七行 if 和 while 都能过，但是while 时间长。我不确定if的写法完不完备。

```
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        if(nums.size()==0) return -1;

        for(int i=0; i<nums.size(); i++){
            if(nums[i]!= i && nums[i]>=0 && nums[i] <nums.size() && nums[nums[i]] != nums[i]){
                swap(nums[i], nums[nums[i]]);
            }
        }

        for(int i=0; i<nums.size(); i++){
            if(nums[i] != i) return i;
        }
        return nums.size();
    }
};
```



### 剑指 Offer 46. 把数字翻译成字符串

动态规划。注意边界

```
class Solution {
public:
    int translateNum(int num) {
        if(num < 0) return 0;
        if(num == 0) return 1;
        vector<int>number;
        while(num){
            number.push_back(num%10);
            num/=10;
        }
        reverse(number.begin(), number.end());
        int n = number.size();

        int dp[n];
        memset(dp, 0, sizeof(dp));
        dp[0] = 1;
        for(int i=1; i<n; i++){
            dp[i] = dp[i-1];
            if(number[i-1] && number[i-1]*10 + number[i] <=25){
                if(i==1) dp[i]++;
                else dp[i] += dp[i-2];
            }
        }
        return dp[n-1];
    }
};
```





### 剑指 Offer 12. 矩阵中的路径

常规dfs

```
class Solution {
public:

    int dir[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    int m, n;
    bool visited[8][8];

    bool dfs(vector<vector<char>>& board, string& word, int x, int y, int pos){
        
        if(board[x][y] != word[pos])
            return false;
        if(pos == word.size()-1)  // 不能等到 pos == word.size()，如果最后一个位置在边界，下一个位置newx newy会过滤掉
            return true;

        for(int i=0; i<4; i++){
            int newx = x + dir[i][0];
            int newy = y + dir[i][1];
            if(newx <0 || newx >=m || newy<0 || newy >=n || visited[newx][newy])
                continue;
            
            visited[newx][newy] = true;
            if(dfs(board, word, newx, newy, pos+1))
                return true;
            visited[newx][newy] = false;
        }
        return false;
    }

    bool exist(vector<vector<char>>& board, string word) {
        if(board.size() == 0 || board[0].size()==0 || word.size() == 0) return false;

        m = board.size();
        n = board[0].size();
        
        memset(visited, false, sizeof(visited));
    
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(board[i][j] == word[0]){  // 嗯最好还是在dfs外部找起始点
                    visited[i][j] = true;
                    if(dfs(board, word, i, j, 0))
                        return true;
                    visited[i][j] = false;
                }
                
            }
        }
        return false;
        
    }
};
```

