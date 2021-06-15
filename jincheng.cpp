#include<cstdio>
#include<cstring>
#include<queue>
#include<iostream>
#include<algorithm>
using namespace std;
struct node
{
    int num;            ///序号
    char name[10];      ///进程名
    int Gen_time;       ///产生时间
    double Ser_time;    ///要求服务时间
    int Pty;            ///优先级priority
    char vis;           ///进程状态R表示就绪，E表示结束
    int ends;           ///完成时间
    double t_time;      ///周转时间
    double rt_time;     ///带权周转时间
} p[1000],p1[1000],
p2[1000],p3[1000],p4[1000];
///p1可抢占式的进程数据，p2不可抢占式的进程数据
///p3短进程优先调度的进程数据，p4先来先服务的进程数据
int number;         ///出入的进程个数
void Init(node m[],node n[])    ///初始化基本数据
{
    for(int i=0; i<number; i++)
    {
        m[i].num=n[i].num;
        strcpy(m[i].name,n[i].name);
        m[i].Gen_time=n[i].Gen_time;
        m[i].Ser_time=n[i].Ser_time;
        m[i].Pty=n[i].Pty;
        m[i].vis=n[i].vis;
    }
}
void print(int by_order[],int k,node q[])   ///输出进程执行结果
{
    printf("%-16s%-16s%-16s%-16s%-16s%-16s%-16s\n",
           "进程序列号","进程名","产生时间","要求服务时间","完成时间",
           "周转时间","带权周转时间");
    double sum1=0,sum2=0;
    for(int i=0; i<number; i++)
    {
        printf("%-8d\t%-8s\t%-8d\t%-8.2lf\t%-8d\t%-8.2lf\t%-8.2lf\n",
               q[i].num,q[i].name,q[i].Gen_time,q[i].Ser_time,q[i].ends,
               q[i].t_time,q[i].rt_time);
        sum1+=q[i].t_time;
        sum2+=q[i].rt_time;
    }
    cout<<endl;
    printf("平均周转时间为：%.2lf 平均带权周转时间为：%.2lf\n\n",sum1/number,sum2/number);
    printf("各进程执行顺序为：\n");
    for(int i=0; i<k; i++)
    {
        if(i==0)
            printf("%d ",by_order[i]);
        else if(i!=0&&by_order[i]!=by_order[i-1])
            printf("%d ",by_order[i]);
    }
    printf("\n");
    return;
}
bool cmp(node a,node b)  ///按照进程的序号排列
{
    return a.num<b.num;
}

bool priority(node a,node b)    ///按照进程的优先级排序
{
    return a.Pty<b.Pty;///数字小的优先级高
}
void nopreemptible_priority() ///不可抢占式优先级法
{
    Init(p1,p);     ///初始化
    sort(p1,p1+number,priority);       ///各进程按照优先级排序
    int run_sum=0;  ///进程队列执行总时间
    int i=0;///进程执行的时刻
    int by_order[number];
    int k=0;
    memset(by_order,0,sizeof(by_order));
    queue<node>Q,Q1;///Q存放执行进程，Q1存放执行完的进程
    for(int h=0; h<number; h++) ///寻找第一个符合执行条件的进程
    {
        if(p1[h].Gen_time<=i)
        {
            p1[h].vis='E';
            by_order[k++]=p1[h].num;
            Q.push(p1[h]);
            break;
        }
    }
    while(!Q.empty())
    {
        node q;
        q=Q.front();
        Q.pop();
        if(q.Gen_time<=i)   ///该时刻进程已经产生
        {
            run_sum+=q.Ser_time;
            q.t_time=run_sum-q.Gen_time;    ///周转时间
            q.rt_time=q.t_time/q.Ser_time;  ///带权周转时间
            q.ends=run_sum;     ///结束时间
        }
        i=run_sum;
        Q1.push(q);     ///放入进程执行完成的队列
        for(int j=0; j<number; j++) ///选择下一个需要执行的进程
        {
            if(p1[j].Gen_time<=i&&p1[j].vis=='R')
            {   ///进程已经产生且没有执行完成
                p1[j].vis='E';
                by_order[k++]=p1[j].num;
                Q.push(p1[j]);
                break;
            }
        }
    }
    int cnt=0;
    while(!Q1.empty())
    {
        p1[cnt++]=Q1.front();
        Q1.pop();
    }
    sort(p1,p1+number,cmp);     ///将各进程按照编号排序
    printf("\t\t\t------不可抢占式优先级法进程调度执行结果-----\n\n");
    print(by_order,k,p1);
    return;
}

void preemptible_priority()   ///可抢占式优先级法
{
    Init(p2,p);   ///初始化
    sort(p2,p2+number,priority);       ///各进程按照优先级排序

    int time[1000];
    memset(time,-1,sizeof(time));
    int sum=0;
    for(int i=0; i<number; i++)
    {
        time[p2[i].Gen_time]=i; ///标记按照优先级排好顺序后的位置
        sum+=p2[i].Ser_time;
    }
    int by_order[sum];
    int k=0;
    memset(by_order,0,sizeof(by_order));
    int i=0,j;
    while(i<sum)
    {
        for(j=0; j<number; j++)
        {
            ///当前按照优先级降序排列好的进程队列
            if(p2[j].Gen_time<=i&&p2[j].vis=='R')
                break;
        }
        while(1)
        {
            //cout<<"i="<<i<<endl;cout<<"j="<<p2[j].num<<endl;
            i++;    ///运行时间增加
            p2[j].Ser_time--;   ///要求服务时间减少
            by_order[k++]=p2[j].num;    ///记录当前进程的序号
            if(time[i]!=-1&&p2[j].Pty>p2[time[i]].Pty)
            {
                ///该时刻有比该进程优先级高的进程产生，则抢占
                break;
            }
            if(p2[j].Ser_time==0) ///进程执行结束的处理
            {
                p2[j].ends=i;
                p2[j].vis='E';
                p2[j].t_time=i-p2[j].Gen_time;  ///周转时间
                break;
            }
        }
    }
    sort(p2,p2+number,cmp); ///按照进程序号排序
    for(int i=0; i<number; i++)
    {
        p2[i].Ser_time=p[i].Ser_time;
        p2[i].rt_time=p2[i].t_time/p2[i].Ser_time;  ///带权周转时间
    }
    printf("\t\t\t------可抢占式优先级法进程调度执行结果-----\n\n");
    print(by_order,k,p2);
    return;
}
bool Short(node a,node b)   ///按照进程的要求服务时间升序排列
{
    return a.Ser_time<b.Ser_time;
}
void Deal(node p3[],int by_order[])
{
    int run_sum=0;  ///进程队列执行总时间
    int i=0;///进程执行的时刻
    //int by_order[number];
    int k=0;
    memset(by_order,0,sizeof(by_order));
    while(1)
    {
        int flag=0;///标记此次循环是否有进程被处理
        for(int j=0; j<number; j++)
        {
            if(p3[j].Gen_time<=i&&p3[j].vis=='R')
            {
                flag=1;
                by_order[k++]=p3[j].num;    ///记录进程被执行的顺序
                run_sum+=p3[j].Ser_time;
                p3[j].vis='E';
                i+=p3[j].Ser_time;
                p3[j].ends=run_sum;     ///结束时间
                p3[j].t_time=run_sum-p3[j].Gen_time;        ///周转时间
                p3[j].rt_time=p3[j].t_time/p3[j].Ser_time;  ///带权周转时间
                break;
            }
        }
        if(!flag) ///所有进程都处理完毕，退出
            break;
    }
    sort(p3,p3+number,cmp);     ///将各进程按照编号排序

    return;
}
void Short_process() //短进程优先算法
{
    Init(p3,p);
    sort(p3,p3+number,Short);///按照要服务时间升序排列
    int by_order[number];
    Deal(p3,by_order);
    printf("\t\t\t------短进程优先级法进程调度执行结果-----\n\n");
    print(by_order,number,p3);///输出结果
}
bool H_R(node a,node b) ///按照优先权降序排序
{
    return a.rt_time>b.rt_time;
}
bool cmp_G(node a,node b)
{
    return a.Gen_time<b.Gen_time;
}
void FCFS() ///先来先服务算法
{
    Init(p4,p); ///初始化
    sort(p4,p4+number, cmp_G); ///按照产生时间的先后顺序排队
    int run_sum=0;
    int by_order[number];
    for(int i=0;i<number;i++)
    {
        by_order[i]=p4[i].num;
        p4[i].vis='E';
        p4[i].ends=run_sum;
        run_sum+=p4[i].Ser_time;
        p4[i].t_time=run_sum-p4[i].Gen_time;
        p4[i].rt_time=p4[i].t_time/p4[i].Ser_time;
    }
    printf("\t\t\t------先来先服务进程调度执行结果-----\n\n");
    print(by_order,number,p4);///输出结果
    return;
}
void Save_data()
{
    printf("\t\t\t-----进程列表-----\n\n");
    printf("%-16s%-16s%-16s%-16s%-16s%-16s\n",
           "进程序列号","进程名","产生时间","要求服务时间","优先级","状态");
    for(int i=0; i<number; i++)
    {
        printf("%-8d\t%-8s\t%-8d\t%-8.2lf\t%-8d\t%-8c\t\n",
               p[i].num,p[i].name,p[i].Gen_time,p[i].Ser_time,p[i].Pty,p[i].vis);
    }
    cout<<endl;
    return;
}
int main()
{
    ///freopen("in.txt","r",stdin);
    printf("输入进程的个数：\n");
    cin>>number;
    memset(p,0,sizeof(p));
    for(int i=0; i<number; i++)
    {
        printf("请输入第%d个进程的进程的相关属性：\n",i+1);
        printf("请输入进程名：");
        cin>>p[i].name;///cout<<endl;
        printf("请输入进程产生时间：");
        cin>>p[i].Gen_time;//cout<<endl;
        printf("请输入进程要求服务的时间：");
        cin>>p[i].Ser_time;
        getchar();
        printf("请输入进程的优先级：");
        cin>>p[i].Pty;//cout<<endl;
        p[i].num=i+1;
        p[i].vis='R';
    }
    char e,str;
    while(1)
    {
        getchar();
        str=getchar();
        if(str=='\n')
            system("cls");      ///清除上述输出
        Save_data();
        printf("\t\t********************Menu********************\n\n");
        printf("\t\t\t1.不可抢占式优先级法\n\n");
        printf("\t\t\t2.可抢占式优先级法\n\n");
        printf("\t\t\t3.短进程优先法\n\n");
        printf("\t\t\t4.先来先服务算法\n\n");
        printf("\t\t\t输入“#”退出程序\n\n");
        printf("\t\t********************************************\n\n");
        printf("\t\t请输入指令：");
        cin>>e;
        if(e=='#') break;
        switch(e)
        {
        case '1':
            nopreemptible_priority();   ///不可抢占式优先级法
            break;
        case '2':
            preemptible_priority();     ///可抢占式优先级法
            break;
        case '3':
            Short_process();   ///短进程优先法
            break;
        case '4':
            FCFS();           ///先来先服务算法
            break;
        }
    }
    return 0;
}
