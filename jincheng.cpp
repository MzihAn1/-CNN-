#include<cstdio>
#include<cstring>
#include<queue>
#include<iostream>
#include<algorithm>
using namespace std;
struct node
{
    int num;            ///���
    char name[10];      ///������
    int Gen_time;       ///����ʱ��
    double Ser_time;    ///Ҫ�����ʱ��
    int Pty;            ///���ȼ�priority
    char vis;           ///����״̬R��ʾ������E��ʾ����
    int ends;           ///���ʱ��
    double t_time;      ///��תʱ��
    double rt_time;     ///��Ȩ��תʱ��
} p[1000],p1[1000],
p2[1000],p3[1000],p4[1000];
///p1����ռʽ�Ľ������ݣ�p2������ռʽ�Ľ�������
///p3�̽������ȵ��ȵĽ������ݣ�p4�����ȷ���Ľ�������
int number;         ///����Ľ��̸���
void Init(node m[],node n[])    ///��ʼ����������
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
void print(int by_order[],int k,node q[])   ///�������ִ�н��
{
    printf("%-16s%-16s%-16s%-16s%-16s%-16s%-16s\n",
           "�������к�","������","����ʱ��","Ҫ�����ʱ��","���ʱ��",
           "��תʱ��","��Ȩ��תʱ��");
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
    printf("ƽ����תʱ��Ϊ��%.2lf ƽ����Ȩ��תʱ��Ϊ��%.2lf\n\n",sum1/number,sum2/number);
    printf("������ִ��˳��Ϊ��\n");
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
bool cmp(node a,node b)  ///���ս��̵��������
{
    return a.num<b.num;
}

bool priority(node a,node b)    ///���ս��̵����ȼ�����
{
    return a.Pty<b.Pty;///����С�����ȼ���
}
void nopreemptible_priority() ///������ռʽ���ȼ���
{
    Init(p1,p);     ///��ʼ��
    sort(p1,p1+number,priority);       ///�����̰������ȼ�����
    int run_sum=0;  ///���̶���ִ����ʱ��
    int i=0;///����ִ�е�ʱ��
    int by_order[number];
    int k=0;
    memset(by_order,0,sizeof(by_order));
    queue<node>Q,Q1;///Q���ִ�н��̣�Q1���ִ����Ľ���
    for(int h=0; h<number; h++) ///Ѱ�ҵ�һ������ִ�������Ľ���
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
        if(q.Gen_time<=i)   ///��ʱ�̽����Ѿ�����
        {
            run_sum+=q.Ser_time;
            q.t_time=run_sum-q.Gen_time;    ///��תʱ��
            q.rt_time=q.t_time/q.Ser_time;  ///��Ȩ��תʱ��
            q.ends=run_sum;     ///����ʱ��
        }
        i=run_sum;
        Q1.push(q);     ///�������ִ����ɵĶ���
        for(int j=0; j<number; j++) ///ѡ����һ����Ҫִ�еĽ���
        {
            if(p1[j].Gen_time<=i&&p1[j].vis=='R')
            {   ///�����Ѿ�������û��ִ�����
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
    sort(p1,p1+number,cmp);     ///�������̰��ձ������
    printf("\t\t\t------������ռʽ���ȼ������̵���ִ�н��-----\n\n");
    print(by_order,k,p1);
    return;
}

void preemptible_priority()   ///����ռʽ���ȼ���
{
    Init(p2,p);   ///��ʼ��
    sort(p2,p2+number,priority);       ///�����̰������ȼ�����

    int time[1000];
    memset(time,-1,sizeof(time));
    int sum=0;
    for(int i=0; i<number; i++)
    {
        time[p2[i].Gen_time]=i; ///��ǰ������ȼ��ź�˳����λ��
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
            ///��ǰ�������ȼ��������кõĽ��̶���
            if(p2[j].Gen_time<=i&&p2[j].vis=='R')
                break;
        }
        while(1)
        {
            //cout<<"i="<<i<<endl;cout<<"j="<<p2[j].num<<endl;
            i++;    ///����ʱ������
            p2[j].Ser_time--;   ///Ҫ�����ʱ�����
            by_order[k++]=p2[j].num;    ///��¼��ǰ���̵����
            if(time[i]!=-1&&p2[j].Pty>p2[time[i]].Pty)
            {
                ///��ʱ���бȸý������ȼ��ߵĽ��̲���������ռ
                break;
            }
            if(p2[j].Ser_time==0) ///����ִ�н����Ĵ���
            {
                p2[j].ends=i;
                p2[j].vis='E';
                p2[j].t_time=i-p2[j].Gen_time;  ///��תʱ��
                break;
            }
        }
    }
    sort(p2,p2+number,cmp); ///���ս����������
    for(int i=0; i<number; i++)
    {
        p2[i].Ser_time=p[i].Ser_time;
        p2[i].rt_time=p2[i].t_time/p2[i].Ser_time;  ///��Ȩ��תʱ��
    }
    printf("\t\t\t------����ռʽ���ȼ������̵���ִ�н��-----\n\n");
    print(by_order,k,p2);
    return;
}
bool Short(node a,node b)   ///���ս��̵�Ҫ�����ʱ����������
{
    return a.Ser_time<b.Ser_time;
}
void Deal(node p3[],int by_order[])
{
    int run_sum=0;  ///���̶���ִ����ʱ��
    int i=0;///����ִ�е�ʱ��
    //int by_order[number];
    int k=0;
    memset(by_order,0,sizeof(by_order));
    while(1)
    {
        int flag=0;///��Ǵ˴�ѭ���Ƿ��н��̱�����
        for(int j=0; j<number; j++)
        {
            if(p3[j].Gen_time<=i&&p3[j].vis=='R')
            {
                flag=1;
                by_order[k++]=p3[j].num;    ///��¼���̱�ִ�е�˳��
                run_sum+=p3[j].Ser_time;
                p3[j].vis='E';
                i+=p3[j].Ser_time;
                p3[j].ends=run_sum;     ///����ʱ��
                p3[j].t_time=run_sum-p3[j].Gen_time;        ///��תʱ��
                p3[j].rt_time=p3[j].t_time/p3[j].Ser_time;  ///��Ȩ��תʱ��
                break;
            }
        }
        if(!flag) ///���н��̶�������ϣ��˳�
            break;
    }
    sort(p3,p3+number,cmp);     ///�������̰��ձ������

    return;
}
void Short_process() //�̽��������㷨
{
    Init(p3,p);
    sort(p3,p3+number,Short);///����Ҫ����ʱ����������
    int by_order[number];
    Deal(p3,by_order);
    printf("\t\t\t------�̽������ȼ������̵���ִ�н��-----\n\n");
    print(by_order,number,p3);///������
}
bool H_R(node a,node b) ///��������Ȩ��������
{
    return a.rt_time>b.rt_time;
}
bool cmp_G(node a,node b)
{
    return a.Gen_time<b.Gen_time;
}
void FCFS() ///�����ȷ����㷨
{
    Init(p4,p); ///��ʼ��
    sort(p4,p4+number, cmp_G); ///���ղ���ʱ����Ⱥ�˳���Ŷ�
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
    printf("\t\t\t------�����ȷ�����̵���ִ�н��-----\n\n");
    print(by_order,number,p4);///������
    return;
}
void Save_data()
{
    printf("\t\t\t-----�����б�-----\n\n");
    printf("%-16s%-16s%-16s%-16s%-16s%-16s\n",
           "�������к�","������","����ʱ��","Ҫ�����ʱ��","���ȼ�","״̬");
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
    printf("������̵ĸ�����\n");
    cin>>number;
    memset(p,0,sizeof(p));
    for(int i=0; i<number; i++)
    {
        printf("�������%d�����̵Ľ��̵�������ԣ�\n",i+1);
        printf("�������������");
        cin>>p[i].name;///cout<<endl;
        printf("��������̲���ʱ�䣺");
        cin>>p[i].Gen_time;//cout<<endl;
        printf("���������Ҫ������ʱ�䣺");
        cin>>p[i].Ser_time;
        getchar();
        printf("��������̵����ȼ���");
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
            system("cls");      ///����������
        Save_data();
        printf("\t\t********************Menu********************\n\n");
        printf("\t\t\t1.������ռʽ���ȼ���\n\n");
        printf("\t\t\t2.����ռʽ���ȼ���\n\n");
        printf("\t\t\t3.�̽������ȷ�\n\n");
        printf("\t\t\t4.�����ȷ����㷨\n\n");
        printf("\t\t\t���롰#���˳�����\n\n");
        printf("\t\t********************************************\n\n");
        printf("\t\t������ָ�");
        cin>>e;
        if(e=='#') break;
        switch(e)
        {
        case '1':
            nopreemptible_priority();   ///������ռʽ���ȼ���
            break;
        case '2':
            preemptible_priority();     ///����ռʽ���ȼ���
            break;
        case '3':
            Short_process();   ///�̽������ȷ�
            break;
        case '4':
            FCFS();           ///�����ȷ����㷨
            break;
        }
    }
    return 0;
}
