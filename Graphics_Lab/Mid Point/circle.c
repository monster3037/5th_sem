#include<windows.h>

#include<GL\glew.h>

#include<GL\glut.h>

#include <stdio.h>

#include <stdlib.h>

int  x,y,r,xc,yc;

void putpixel(int x, int y)

{

    glPointSize(8.0);

    glColor3f(1.0, 0.0, 0.0);

    glBegin(GL_POINTS);

    glVertex2i(xc + x, yc + y);
    glVertex2i(xc + x, yc - y);

    glVertex2i(xc + y, yc + x);

    glVertex2i(xc + y, yc - x);

    glVertex2i(xc - x, yc - y);

    glVertex2i(xc - y, yc - x);

    glVertex2i(xc - x, yc + y);

    glVertex2i(xc - y, yc + x);

    glEnd();

}

void display()

{

    glColor3f(1.0, 0.0, 0.0);  //Quadrant Plot Graph

    glBegin(GL_LINES);

    glVertex2i(-50, 0);

    glVertex2i(50, 0);

    glVertex2i(0, -50);

    glVertex2i(0, 50);

    glEnd();

    int d[r];
    d[0]=1-r;
    x=0,y=0;
    y=r;
    if(d[0]<=0)
    {

        putpixel(x,y);
        d[1]=d[0]+2*x+1;
        x=x+1;
    }
    else
    {

        putpixel(x,y);
        d[1]=d[0]+2*x+3-2*y;
        x=x+1;
        y=y-1;
    }
    int i=1;
    for(; i<y; i++)
    {
        if(d[i]<=0)
        {
            putpixel(x,y);
            d[i+1]=d[i]+2*x+1;
            x=x+1;
        }
        else
        {
            putpixel(x,y);
            d[i+1]=d[i]+2*x+3-2*y;
            x=x+1;
            y=y-1;
        }
    }

    glEnd();

    glFlush();

}

void init()

{

    glClearColor(0.7, 0.7, 0.7, 0.7);

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    gluOrtho2D(-50, 50, -50, 50);

}

int main(int argc, char* argv[])
{

    printf("Dhruv Singhal || 500075346 \n");
    printf("Enter the coordinates of the circle's centre:");

    scanf("%d  %d",&xc,&yc);

    printf("Enter the value of r : ");

    scanf("%d",&r);

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutInitWindowSize(350, 350);

    glutInitWindowPosition(100, 100);

    glutCreateWindow("Midpoint Circle [Dhruv Singhal || 500075346]");

    init();

    glutDisplayFunc(display);

    glutMainLoop();

}
