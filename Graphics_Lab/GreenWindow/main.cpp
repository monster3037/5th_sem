#include<windows.h>
#include<GL/glu.h>
#include<GL/glut.h>
#include<iostream>

using namespace std;

void display()
{
glClear(GL_COLOR_BUFFER_BIT);
glFlush();
}

void init()
{
    glClearColor(0.0,1,0.0,1.0);
    glColor3f(0.0,1.0,0.0);

}

int main(int argc , char** argv)
{
 glutInit(&argc,argv);
 glutInitDisplayMode(GLUT_RGB);
 glutInitWindowPosition(200,100);
 glutInitWindowSize(500,500);
 glutCreateWindow("Green Window (Jay Gupta || 500076426)");
 glutDisplayFunc(display);
 init();
 glutMainLoop();
 return 0;
}
