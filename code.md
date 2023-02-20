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