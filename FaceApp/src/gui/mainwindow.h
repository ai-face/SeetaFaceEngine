#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>
#include <QDebug>
#include <QDir>
#include <qlistwidget.h>

#include <qfilesystemmodel.h>
#include <vector>
#include <string>


#include <searcher/SearchActor.h>
#include <searcher/FaceEngine.h>

#include<QMenu>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    static const int PROCESS_TIMEOUT = 50;

private:
    Ui::MainWindow *ui;
    void initForm();

    FaceEngine engine;

    int MinFaceSize;
    float ImagePyramidScaleFactor;

    unsigned int numKNN;
    unsigned int numReranking;

    QListWidget *imgs_listeWidget;

private slots:
    void on_listView_clicked(const QModelIndex &index);
    void on_listView_doubleClicked(const QModelIndex &index);
    void on_faceDetectionButton_clicked(bool checked);
    void on_queryButton_clicked();
    void on_ImgsOpenButton_clicked();

protected:

    bool eventFilter(QObject *watched, QEvent *event) override;
    void dropEvent(QDropEvent * event) override;

public:


    SearchActor actor;

    //======================
    QDir rootdir;
    QDir modeldir;
    QDir cropsdir;
    QDir srcdir;
    void setCropsDir(const QString & dir);
    void clearDb();
    void loadDb();
    void createDb(const QString & srcdir);
    void searchSimilearImgs(const QString & target);
};

#endif // MAINWINDOW_H
