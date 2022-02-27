#include<windows.h>
#include<GL/glu.h>
#include<GL/glut.h>
#include<stdio.h>

float x1,x2,y1,y2;

void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0, 0.0, 0.0);  //Quadrant Plot Graph

    glBegin(GL_LINES);

    glVertex2i(-50, 0);

    glVertex2i(50, 0);

    glVertex2i(0, -50);

    glVertex2i(0, 50);

    glEnd();
    float dy,dx,step,x,y,k,m;
    dx=x2-x1;
    dy=y2-y1;
    m=dy/dx;

    if(abs(dx)> abs(dy))
    {
        step = abs(dx);
    }
    else
    {
        step = abs(dy);
    }

    x=x1;
    y=y1;
    glBegin(GL_POINTS);

    glVertex2i(x,y);
    glEnd();

    for (k=1 ; k<=step; k++)
    { // 0.5 factor is added to remove the stair case effect
        if(m<1)
        {
            x= 1*0.5 + x;
            y= m*0.5 + y;
        }
        if(m==1)
        {
            x= 1*0.5 + x;
            y= 1*0.5 + y;
        }
        if(m>1)
        {
            x= (1/m)*0.5 + x;
            y= 1*0.5 + y;
        }
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);
        glVertex2f(x,y);
        glEnd();
    }

    glFlush();
}

void init(void)
{
    glClearColor(1.0,1.0,0.0,0.0);

    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(-50,50,-50,50);
}

int main(int argc, char** argv)
{
    printf("Enter the value of x1 : ");
    scanf("%f",&x1);
    printf("Enter the value of y1 : ");
    scanf("%f",&y1);
    printf("Enter the value of x2 : ");
    scanf("%f",&x2);
    printf("Enter the value of y2 : ");
    scanf("%f",&y2);

    glutInit(&argc, argv);
    glutInitDisplayMode ( GLUT_RGB);
    glutInitWindowSize (500, 500);
    glutInitWindowPosition (100,100);
    glutCreateWindow ("DDA Line Algo [Dhruv Singhal || 500075346]");
    init();
    glutDisplayFunc(display);
    glutMainLoop();

    return 0;
}
