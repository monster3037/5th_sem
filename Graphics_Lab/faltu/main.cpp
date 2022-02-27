#include <math.h>
#include <gl/glut.h>
#include <stdlib.h>
#include<iostream>
#include<conio.h>

using namespace std;

GLint xc,yc,r;

struct Color {
    GLfloat r;
    GLfloat g;
    GLfloat b;
};


Color getPixelColor(GLint x, GLint y) {
    Color color;
    glReadPixels(x, y, 1, 1, GL_RGB, GL_FLOAT, &color);
    return color;
}

void setPixelColor(GLint x, GLint y, Color color) {
    glColor3f(color.r, color.g, color.b);
    glBegin(GL_POINTS);
    glVertex2i(x, y);
    glEnd();
    glFlush();
}

void floodFill(GLint x, GLint y, Color oldColor, Color newColor) {
    Color color;
    color = getPixelColor(x, y);

    if(color.r == oldColor.r && color.g == oldColor.g && color.b == oldColor.b)
    {
        setPixelColor(x, y, newColor);
        floodFill(x+1, y, oldColor, newColor);
        floodFill(x, y+1, oldColor, newColor);
        floodFill(x-1, y, oldColor, newColor);
        floodFill(x, y-1, oldColor, newColor);
    }
}


void plotSymmetric(float x,float y){

    Color color;
    color.r=0;
    color.g=0;
    color.b=0;



    setPixelColor(xc+x,yc+y,color);
    setPixelColor(xc-x,yc+y,color);
    setPixelColor(xc-x,yc-y,color);
    setPixelColor(xc+x,yc-y,color);

    setPixelColor(xc+y,yc+x,color);
    setPixelColor(xc-y,yc+x,color);
    setPixelColor(xc-y,yc-x,color);
    setPixelColor(xc+y,yc-x,color);


}

void draw_circle() {
    GLfloat x,y,p;


    x=0;
    y=r;

    p = 1-r;

    plotSymmetric(x,y);

    while(x<y){

       x++;

       if(p < 0){
            p += 2*x + 1;
       }
       else{
            y--;
            p += 2*x +1 - 2*y;
       }
       plotSymmetric(x,y);

    }
}

void display(void) {

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
        draw_circle();
    glEnd();

    Color newColor = {1.0f, 0.0f, 0.0f};
    Color oldColor = {1.0f, 1.0f, 1.0f};

    floodFill(xc,yc, oldColor, newColor);

    glFlush();
}

void init() {
    glClearColor(1.0, 1.0, 1.0, 0.0);
    glColor3f(0.0, 0.0, 0.0);
    glPointSize(1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 500, 0, 500);


}

int main(int argc, char** argv)
{

    cout<<"Enter center coordinates (xc yc) : ";
    cin>>xc>>yc;
    cout<<"Enter radius : ";
    cin>>r;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(200, 200);
    glutCreateWindow("FloodFill");
    init();
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
