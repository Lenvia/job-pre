
sql运行顺序 
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



### SQL22 统计每个学校的答过题的用户的平均答题数

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



