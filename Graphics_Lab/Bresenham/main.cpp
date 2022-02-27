#include<windows.h>
#include<GL/glut.h>
#include<stdio.h>
int xa,xb,ya,yb;
void display (void)
{
    int dx=xb-xa;
    int dy=yb-ya;
    int x=xa,y=ya;
    glClear (GL_COLOR_BUFFER_BIT);
    glColor3f (0.0, 1.0, 0.0);
    glBegin(GL_POINTS);
    glVertex2i(x,y);
    int m;
    m=dy/dx;
    if(m<1)
    {
        int p0 = 2*dy - dx;
        int p =p0;
        int k;
        for(k=0; k<dx; k++)
        {
            if(p<0)
            {
                x = x+1;
                glVertex2i(x,y);
                p=p+2*dy;
                printf("%d %d\n",x,y);
            }
            else
            {
                x = x+1;
                y = y+1;
                glVertex2i(x,y);
                p=p+2*dy-2*dx;
                printf("%d %d\n",x,y);
            }
        }
    }
    if(m>=1)
    {
        int P0 = 2*dx - dy;
        int P =P0;
        int K;
        for(K=0; K<dy; K++)
        {
            if(P<0)
            {
                y = y+1;
                glVertex2i(x,y);
                P=P+(2*dx);
                printf("%d %d\n",x,y);
            }
            else
            {
                x = x+1;
                y = y+1;
                glVertex2i(x,y);
                P=P+2*dx-2*dy;
                printf("%d %d\n",x,y);
            }
        }
    }
    glEnd();
    glFlush();
}
void init(void)
{
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-100,100,-100,100);
}
int main(int argc, char** argv)
{
    printf("Enter the value of x1 : ");
    scanf("%d",&xa);
    printf("Enter the value of y1 : ");
    scanf("%d",&ya);
    printf("Enter the value of x2 : ");
    scanf("%d",&xb);
    printf("Enter the value of y2 : ");
    scanf("%d",&yb);
    glutInit(&argc, argv);
    glutInitDisplayMode (GLUT_RGB);
    glutInitWindowSize (500, 500);
    glutInitWindowPosition (100, 100);
    glutCreateWindow ("Bresenham Line Algorithm [Dhruv Singhal || 500075346]");
    init();
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
