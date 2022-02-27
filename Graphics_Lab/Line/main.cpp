#include<windows.h>
#include<GL/glut.h>
#include<math.h>
#include<iostream>
using namespace std;
void display()
{
    glClearColor(1.0,0.0,0.0,0.0);
    glColor3f(1,1,1);
    glClear(GL_COLOR_BUFFER_BIT);
    float x1=-0.3,y1=-0.5;
    float x2=0.2,y2=0.5;
    float m,c,y;
    m=(y2-y1)/(x2-x1);
    c=y1-m*x1;
    glBegin(GL_LINES);
    for( float x=x1; x<=x2; x=x+0.002)
    {
        y=m*x+c;
        glVertex2f(x,y);
    }
    glEnd();
    cout<<"m="<<m;
    glFlush();
}
int main(int argc, char** argv)
{
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_SINGLE |GLUT_RGB);
    glutInitWindowPosition(200,100);
    glutInitWindowSize(500,500);
    glutCreateWindow("Line [Arihant Jain || 500075953]");
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
