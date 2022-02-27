#include<windows.h>
#include <GL/glut.h>
int ww = 600, wh = 700; // window width and height
float fillCol[3] = {0.4,0.0,0.0};
float borderCol[3] = {0.0,0.0,0.0};
void setPixel(int pointx, int pointy, float f[3])
{
    glBegin(GL_POINTS);
    glColor3fv(f);
    glVertex2i(pointx,pointy);
    glEnd();
    glFlush();
}
void getPixel(int x, int y, float pixels[3])
{
    glReadPixels(x,y,1.0,1.0,GL_RGB,GL_FLOAT,pixels);
}
void drawPoly(int x1, int y1, int x2, int y2)
{
    glColor3f(0.0,1.0,0.0);
    glBegin(GL_LINES);
    glVertex2i(x1, y1);
    glVertex2i(x1, y2);
    glVertex2i(x2, y1);
    glVertex2i(x2, y2);
    glVertex2i(x1, y1);
    glVertex2i(x2, y1);
    glVertex2i(x1, y2);
    glVertex2i(x2, y2);
    glEnd();
    glFlush();
}
void display()
{
    glClearColor(1.0,1.0,1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    drawPoly(150,250,200,300);
    glFlush();
}
void boundaryFill4(int x,int y,float fillColor[3],float borderColor[3])
{
    float interiorColor[3];
    getPixel(x,y,interiorColor);
    if((interiorColor[0]!=borderColor[0] && (interiorColor[1])!=borderColor[1] && (interiorColor[2])!=borderColor[2]) && (interiorColor[0]!=fillColor[0] && (interiorColor[1])!=fillColor[1] && (interiorColor[2])!=fillColor[2]))
    {
        setPixel(x,y,fillColor);
        boundaryFill4(x+1,y,fillColor,borderColor);
        boundaryFill4(x-1,y,fillColor,borderColor);
        boundaryFill4(x,y+1,fillColor,borderColor);
        boundaryFill4(x,y-1,fillColor,borderColor);
    }
}
void mouse(int btn, int state, int x, int y)
{
    if(btn==GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        int xi = x;
        int yi = (wh-y);
        boundaryFill4(xi,yi,fillCol,borderCol);
    }
}
void init()
{
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0,(GLdouble)ww,0.0,(GLdouble)wh);
    glMatrixMode(GL_PROJECTION);
}
int main(int argc, char** argv)
{
    glutInit(&argc,argv);
    glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(ww,wh);
    glutCreateWindow("Boundry-Fill [Dhruv Singhal || 500075346 ] ");
    glutDisplayFunc(display);
    init();
    glutMouseFunc(mouse);
    glutMainLoop();
    return 0;
}
