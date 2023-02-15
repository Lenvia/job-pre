## sql执行顺序 

```
- from
- join
- on
- where
- group by
- avg,sum.... （组函数）
- having
- select
- distinct 
- order by
```



## 案例参考

[30道经典SQL面试题讲解(11-20)](https://mp.weixin.qq.com/s?__biz=MzI2MjE3OTA1MA==&mid=2247488048&idx=1&sn=ba490f6c2717e9bed6551b2aec8c08a5&chksm=ea4e43b5dd39caa3b41876535679dca31392aa86006d449b0ce2e7739629fa2039be63864970&scene=21#wechat_redirect)

[30道经典SQL面试题讲解(21-30)](https://mp.weixin.qq.com/s?__biz=MzI2MjE3OTA1MA==&mid=2247489680&idx=1&sn=beffa4b2dbe80ec0d1a50149c4b08fe5&chksm=ea4e4915dd39c0036166745ee152abfd8aa8066f67c909988f537f0b487e1e2b8d812204ba73&scene=27)（偏难）




## 知识点

如果使用了 group by，那select只能出现group by里的字段 或 聚集函数 。

select 最后一个字段不要加点！





## 操作符

### SQL9 查找除复旦大学的用户信息



```
select device_id, gender, age, university
from user_profile
where university not in ("复旦大学")
```

```
select device_id, gender, age, university
from user_profile
where university != "复旦大学"
```



### SQL10 用where过滤空值练习

```
select device_id, gender, age, university
from user_profile
where age is not null
```





### SQL14 操作符混合运用

找到gpa在3.5以上(不包括3.5)的山东大学用户 或 gpa在3.8以上(不包括3.8)的复旦大学同学

```
select device_id, gender, age, university, gpa
from user_profile
where gpa > 3.5 and university = "山东大学"
or gpa > 3.8 and university = "复旦大学"

```

或使用子查询

```
select device_id, gender, age, university, gpa
from user_profile
where
  device_id in(
    select device_id
    from user_profile
    where
      gpa > 3.5 and university = "山东大学"
      or gpa > 3.8 and university = "复旦大学"
  )
```



### SQL15 查看学校名称中含北京的用户

查看所有大学中带有北京的用户的信息

```
select device_id, age, university
from user_profile
where university like "%北京%"
```





## 计算函数

### SQL16 查找GPA最高值

想要知道复旦大学学生gpa最高值是多少



```
select max(gpa) as gpa
from user_profile
where university="复旦大学"
```



```
select gpa
from user_profile
where university = "复旦大学"
order by gpa desc
limit 1
```



### SQL17 计算男生人数以及平均GPA

看一下男性用户有多少人以及他们的平均gpa

问题分解：

1. 限定条件为 男性用户；
2. 有多少人，明显是计数，count函数；
3. 平均gpa，求平均值用avg函数；

```
select count(gender) as male_num, round(avg(gpa), 1) as avg_gpa
from user_profile
where gender = "male"
```



## 分组查询

### SQL18 分组计算练习题

每个学校每种性别的用户数、30天内平均活跃天数和平均发帖数量

问题分解：

- 限定条件：无；
- 每个学校每种性别：按学校和性别分组：`group by gender, university`
- 用户数：count(device_id)
- 30天内平均活跃天数：avg(active_days_within_30)
- 平均发帖数量：avg(question_cnt)

```
select
  gender,
  university,
  count(*) as user_num,
  round(avg(active_days_within_30), 1) as avg_active_day,
  round(avg(question_cnt), 1) as avg_question_cnt
from
  user_profile
group by
  gender,
  university
```



### SQL19 分组过滤练习题

查看每个学校用户的平均发贴和回帖情况，寻找低活跃度学校进行重点运营，请取出平均发贴数低于5的学校或平均回帖数小于20的学校。

问题分解：

- 限定条件：平均发贴数低于5或平均回帖数小于20的学校，`avg(question_cnt)<5 or avg(answer_cnt)<20`，聚合函数结果作为筛选条件时，**不能用where，而是用having语法**，配合重命名即可；
- 按学校输出：需要对每个学校统计其平均发贴数和平均回帖数，因此`group by university`

```
select
  university,
  avg(question_cnt) as avg_question_cnt,
  avg(answer_cnt) as avg_answer_cnt
from
  user_profile
group by
  university
having
  avg_question_cnt < 5
  or avg_answer_cnt < 20
```



### SQL20 分组排序练习题

查看不同大学的用户平均发帖情况，并期望结果按照平均发帖情况进行升序排列

```
select university, avg(question_cnt) as avg_question_cnt
from user_profile
group by university
order by avg_question_cnt asc
```



## 多表查询

### <font color=orange>SQL21 浙江大学用户题目回答情况</font>

查看所有来自浙江大学的用户题目回答明细情况。



问题分解：

- 限定条件：来自浙江大学的用户，学校信息在用户画像表，答题情况在用户练习明细表，因此需要通过device_id关联两个表的数据； 方法1：join两个表，用inner join，条件是`on up.device_id=qpd.device_id and up.university='浙江大学'` 方法2：先从画像表找到浙江大学的所有学生id列表`where university='浙江大学'`，再去练习明细表筛选出id在这个列表的记录，用where in



方法1：inner join

```
select
  qpd.device_id,
  qpd.question_id,
  qpd.result
from
  question_practice_detail as qpd
  inner join user_profile as up on qpd.device_id = up.device_id
  and up.university = "浙江大学"
order by
  question_id
```



方法2：子查询

```
select
  device_id,
  question_id,
  result
from
  question_practice_detail
where
  device_id in (
    select
      device_id
    from
      user_profile
    where
      university = "浙江大学"
  )
  order by question_id
```



### <font color=red>SQL22 统计每个学校的答过题的用户的平均答题数</font>

了解每个学校答过题的用户平均答题数量情况

（说明：某学校用户平均答题数量计算方式为该学校用户答题总次数除以答过题的不同用户个数）



表样例：

**user_profile**

| device_id | gender | age  | university | gpa  | active_days_within_30 |
| --------- | ------ | ---- | ---------- | ---- | --------------------- |
| 2138      | male   | 21   | 北京大学   | 3.4  | 7                     |
| 3214      | male   | NULL | 复旦大学   | 4    | 15                    |
| 6543      | female | 20   | 北京大学   | 3.2  | 12                    |
| 2315      | female | 23   | 浙江大学   | 3.6  | 5                     |

**question_practice_detail**

| device_id | question_id | result |
| --------- | ----------- | ------ |
| 2138      | 111         | wrong  |
| 3214      | 112         | wrong  |
| 3214      | 113         | wrong  |
| 6543      | 111         | right  |



结果样例：

| university | avg_answer_cnt |
| ---------- | -------------- |
| 北京大学   | 1.0000         |
| 复旦大学   | 2.0000         |



问题分解：

- 限定条件：无；
- 每个学校：按学校分组，`group by university`
- 平均答题数量：在每个学校的分组内，用总答题数量除以总人数即可得到平均答题数量`count(question_id) / count(distinct device_id)`。
- 表连接：学校和答题信息在不同的表，需要做连接

```
select
  up.university,
  count(question_id) / count(distinct qpd.device_id) as avg_ans_cnt
from
  user_profile as up
  inner join question_practice_detail as qpd on up.device_id = qpd.device_id
group by
  up.university
```





### <font color=red>SQL23 统计每个学校各难度的用户平均刷题数</font>

计算一些**参加了答题**的不同学校、不同难度的用户平均答题量。

三表联查。

用户信息表：user_profile

| id   | device_id | gender | age  | university | gpa  | active_days_within_30 | question_cnt | answer_cnt |
| ---- | --------- | ------ | ---- | ---------- | ---- | --------------------- | ------------ | ---------- |
| 1    | 2138      | male   | 21   | 北京大学   | 3.4  | 7                     | 2            | 12         |
| 2    | 3214      | male   | NULL | 复旦大学   | 4    | 15                    | 5            | 25         |
| 3    | 6543      | female | 20   | 北京大学   | 3.2  | 12                    | 3            | 30         |

题库练习明细表：question_practice_detail

| id   | device_id | question_id | result |
| ---- | --------- | ----------- | ------ |
| 1    | 2138      | 111         | wrong  |
| 2    | 3214      | 112         | wrong  |
| 3    | 3214      | 113         | wrong  |
| 4    | 6534      | 111         | right  |

表：question_detail

| id   | question_id | difficult_level |
| ---- | ----------- | --------------- |
| 1    | 111         | hard            |
| 2    | 112         | medium          |
| 3    | 113         | easy            |



结果样例：

| university | difficult_level | avg_answer_cnt |
| ---------- | --------------- | -------------- |
| 北京大学   | hard            | 1.0000         |
| 复旦大学   | easy            | 1.0000         |
| 复旦大学   | medium          | 1.0000         |



问题分解：

- 限定条件：无；
- 每个学校：按学校分组`group by university`
- 不同难度：按难度分组`group by difficult_level`
- 平均答题数：总答题数除以总人数`count(qpd.question_id) / count(distinct qpd.device_id)`
- 来自上面信息三个表，需要联表，up与qpd用device_id连接，qd与qpd用question_id连接。



我的答案

```
select
  university,
  difficult_level,
  round(count(qpd.question_id) / count(distinct qpd.device_id), 4) as avg_answer_cnt
from
  user_profile as up
  inner join question_practice_detail as qpd on up.device_id = qpd.device_id
  inner join question_detail as qd on qpd.question_id = qd.question_id
group by
  university,
  difficult_level
```

推荐答案

```
select 
    university,
    difficult_level,
    round(count(qpd.question_id) / count(distinct qpd.device_id), 4) as avg_answer_cnt
from question_practice_detail as qpd

left join user_profile as up
on up.device_id=qpd.device_id

left join question_detail as qd
on qd.question_id=qpd.question_id

group by university, difficult_level
```





### **SQL24** 统计每个用户的平均刷题数

查看**参加了答题**的山东大学的用户在不同难度下的平均答题题目数

（和SQL23的表结构一样）

问题分解：

- 限定条件：山东大学的用户 `up.university="山东大学"`；
- 不同难度：按难度分组`group by difficult_level`
- 平均答题数：总答题数除以总人数count(qpd.question_id) / count(distinct qpd.device_id) 来自上面信息三个表，需要联表，up与qpd用device_id连接并限定大学，qd与qpd用question_id连接。



（group by 的 university 可以去掉）

```
select
  university,
  difficult_level,
  round(
    count(qpd.question_id) / count(distinct qpd.device_id),
    4
  ) as avg_answer_cnt
from
  user_profile as up
  inner join question_practice_detail as qpd on up.device_id = qpd.device_id
  inner join question_detail as qd on qpd.question_id = qd.question_id
where
  university = "山东大学"
group by
  university,
  difficult_level
```



优化一下，把限制条件university放在第一层inner join

```
select
  university,
  difficult_level,
  round(
    count(qpd.question_id) / count(distinct qpd.device_id),
    4
  ) as avg_answer_cnt
from
  user_profile as up
  inner join question_practice_detail as qpd on up.device_id = qpd.device_id and up.university = "山东大学"
  inner join question_detail as qd on qpd.question_id = qd.question_id
group by
  university,
  difficult_level
```





### SQL25 查找山东大学或者性别为男生的信息

分别查看学校为山东大学或者性别为男性的用户的device_id、gender、age和gpa数据，请取出相应结果，结果不去重。

分别查看&结果不去重：所以直接使用两个条件的or是不行的，直接用union也不行，要用union all，分别去查满足条件1的和满足条件2的，然后合在一起不去重。

```
select
  device_id, gender, age, gpa
from
  user_profile
where
  university = "山东大学"
UNION all
select
  device_id, gender, age, gpa
from
  user_profile
where
  gender = "male"
```





## 常用函数

CASE

```
CASE
WHEN 布尔表达式1 THEN 结果表达式1
WHEN 布尔表达式2 THEN 结果表达式2 …
WHEN 布尔表达式n THEN 结果表达式n
[ ELSE 结果表达式n+1 ]
END
```

IF

```
IF(布尔表达式1,结果表达式1,结果表达式2)
```



### <font color=blue>SQL26 计算25岁以上和以下的用户数量</font>

想要将用户划分为25岁以下和25岁及以上两个年龄段，分别查看这两个年龄段用户数量。

方法一：IF

（注意 select 和 if 之间 不用带括号）

```
select
  if (
    age < 25
    OR age is NULL,
    "25岁以下",
    "25岁及以上"
  ) as age_cut,
  count(device_id)
from
  user_profile
group by
  age_cut
```



方法二：CASE

```
select
  CASE
    when age < 25 OR age is NULL then "25岁以下"
    when age >= 25 then "25岁及以上"
  END as age_cut,
  count(device_id)
from
  user_profile
group by
  age_cut
```



### SQL27 查看不同年龄段的用户明细

想要将用户划分为**20岁以下，20-24岁，25岁及以上**三个年龄段

方法一：CASE

```
select
  device_id,
  gender,
  CASE
    when age < 20 then "20岁以下"
    when age <= 24 then "20-24岁"
    when age >= 25 then "25岁及以上"
    else "其他"
  END as age_cut
from
  user_profile
```



方法二：嵌套IF

```
select
  device_id,
  gender,
  if(
    age is NULL,
    "其他",
    if(age < 20, "20岁以下", 
      if(age <= 24, 
        "20-24岁", 
        "25岁及以上"))
  ) as age_cut
from
  user_profile
```



### SQL28 计算用户8月每天的练题数量

计算出2021年8月每天用户练习题目的数量

问题分解：

- 限定条件：2021年8月，写法有很多种，比如用year/month函数的`year(date)=2021 and month(date)=8`，比如用date_format函数的`date_format(date, "%Y-%m")="202108"`
- 每天：按天分组`group by date`
- 题目数量：count(question_id)

```
select
  day(date) as day,
  count(question_id) as question_cnt
from
  question_practice_detail
where month(date) = 8
group by date
```





### <font color=fuchsia>SQL29 计算用户的平均次日留存率</font>

想要查看用户在某天刷题后第二天还会再来刷题的平均概率（占比）。

示例：question_practice_detail

| id   | device_id | quest_id | result | date       |
| ---- | --------- | -------- | ------ | ---------- |
| 1    | 2138      | 111      | wrong  | 2021-05-03 |
| 2    | 3214      | 112      | wrong  | 2021-05-09 |
| 3    | 3214      | 113      | wrong  | 2021-06-15 |
| 4    | 6543      | 111      | right  | 2021-08-13 |
| 5    | 2315      | 115      | right  | 2021-08-13 |
| 6    | 2315      | 116      | right  | 2021-08-14 |
| 7    | 2315      | 117      | wrong  | 2021-08-15 |

<img src="https://tva1.sinaimg.cn/large/008vxvgGgy1h8t16nr3u1j30qc032dg6.jpg" alt="截屏2022-12-05 16.49.32" style="zoom:50%;" />

具体而言，使用两个子查询，查询出两个去重的数据表，并使用条件（q2.date应该是q1.date的后一天）进行筛选。

最后，分别统计q1.device_id 和 q2.device_id 作去重后的所有条目数和去重后的次日留存条目数，即可算出次日留存率。

```sql
select
  count(q2.device_id) / count(q1.device_id) as avg_ret
from
  (
    select
      distinct device_id,
      date
    from
      question_practice_detail
  ) as q1
  left join (
    select
      distinct device_id,
      date
    from
      question_practice_detail
  ) as q2 on q1.device_id = q2.device_id
  and q2.date = DATE_ADD(q1.date, interval 1 day)
```



### SQL30 统计每种性别的人数

文本处理

示例：user_submit

| device_id | profile              | blog_url            |
| --------- | -------------------- | ------------------- |
| 2138      | 180cm,75kg,27,male   | http:/url/bigboy777 |
| 3214      | 165cm,45kg,26,female | http:/url/kittycc   |
| 6543      | 178cm,65kg,25,male   | http:/url/tiger     |
| 4321      | 171cm,55kg,23,female | http:/url/uhksd     |
| 2131      | 168cm,45kg,22,female | http:/urlsydney     |



方法一：SUBSTRING_INDEX

https://www.cnblogs.com/mqxs/p/7380933.html

```
select
  substring_index(profile, ",", -1) as gender,
  count(*)
from
  user_submit
group by
  gender
```



方法二：IF

```
select
  if(profile LIKE '%female', 'female', 'male') as gender,
  count(*) as number
from
  user_submit
group by
  gender
```



### SQL31 提取博客URL中的用户名

（同 SQL30 的表）

常规方法：

```
select
  device_id,
  substring_index(blog_url, "/", -1) as user_name
from
  user_submit;
```

其他方法：

```
select 
-- 替换法 replace(string, '被替换部分','替换后的结果')
-- device_id, replace(blog_url,'http:/url/','') as user_name

-- 截取法 substr(string, start_point, length*可选参数*)
-- device_id, substr(blog_url,11,length(blog_url)-10) as user_nam

-- 删除法 trim('被删除字段' from 列名)
-- device_id, trim('http:/url/' from blog_url) as user_name

-- 字段切割法 substring_index(string, '切割标志', 位置数（负号：从后面开始）)
device_id, substring_index(blog_url,'/',-1) as user_name

from user_submit;
```





### SQL32 截取出年龄

（同 SQL30 的表）

```
select
  substring_index(substring_index(profile, ",", -2),
   ",", 1) as age,
  count(*)
from
  user_submit
group by
  age
```



### <font color=fuchsia>SQL33 找出每个学校GPA最低的同学</font>

想要找到每个学校gpa最低的同学来做调研，请你取出每个学校的最低gpa。

窗口函数：https://zhuanlan.zhihu.com/p/92654574



示例：user_profile

| id   | device_id | gender | age  | university | gpa  | active_days_within_30 | question_cnt | answer_cnt |
| ---- | --------- | ------ | ---- | ---------- | ---- | --------------------- | ------------ | ---------- |
| 1    | 2138      | male   | 21   | 北京大学   | 3.4  | 7                     | 2            | 12         |
| 2    | 3214      | male   |      | 复旦大学   | 4    | 15                    | 5            | 25         |
| 3    | 6543      | female | 20   | 北京大学   | 3.2  | 12                    | 3            | 30         |
| 4    | 2315      | female | 23   | 浙江大学   | 3.6  | 5                     | 1            | 2          |
| 5    | 5432      | male   | 25   | 山东大学   | 3.8  | 20                    | 15           | 70         |
| 6    | 2131      | male   | 28   | 山东大学   | 3.3  | 15                    | 7            | 13         |
| 7    | 4321      | female | 26   | 复旦大学   | 3.6  | 9                     | 6            | 52         |



**【不完善解法：方法一/二，如果有多个人对应同一个最低分，则都会匹配上，是个漏洞】**

方法一：子查询 + inner join

```
select
  up.device_id,
  up.university,
  up.gpa
from
  user_profile as up
  inner join (
    select
      university,
      min(gpa) as mgpa
    from
      user_profile
    group by
      university
  ) as temp 
  on up.university = temp.university
  and up.gpa = temp.mgpa
  order by up.universityselect
  up.device_id,
  up.university,
  up.gpa
from
  user_profile as up
  inner join (
    select
      university,
      min(gpa) as mgpa
    from
      user_profile
    group by
      university
  ) as temp 
  on up.university = temp.university
  and up.gpa = temp.mgpa
  order by up.university
```



方法二：优化一下，inner join 改成了 where(xxx, xxx) in

```
select
  device_id,
  university,
  gpa
from
  user_profile
  where (university, gpa) in (
    select
      university,
      min(gpa)
    from
      user_profile
    group by
      university
  )
  order by university
```



方法三：使用窗口函数（推荐）

```
SELECT
  device_id,
  university,
  gpa
FROM
  (
    SELECT
      device_id,
      university,
      gpa,
      RANK() over (
        PARTITION BY university
        ORDER BY
          gpa
      ) rk
    FROM
      user_profile
  ) a
WHERE
  a.rk = 1;
```







