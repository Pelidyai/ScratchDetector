using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ScratchDetector
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        BackgroundWorker CollectData = new BackgroundWorker();
        BackgroundWorker StatusChange = new BackgroundWorker();


        OpenFileDialog openFileDialog = new OpenFileDialog();
        bool is_work = false;
        string errors = "";
        string outp = "";

        string Im = "";
        public void UrlTextBox_GotMouseCapture(object sender, MouseEventArgs e)
        {
            UrlTextBox.Text = null;
            UrlTextBox.Foreground = System.Windows.Media.Brushes.Gray;
        }
        public MainWindow()
        {
            InitImports();
            CollectData.WorkerReportsProgress = true;
            StatusChange.WorkerReportsProgress = true;
            CollectData.WorkerSupportsCancellation = true;
            StatusChange.WorkerSupportsCancellation = true;

            CollectData.DoWork += new System.ComponentModel.DoWorkEventHandler(CollectData_DoWork);
            StatusChange.DoWork += new System.ComponentModel.DoWorkEventHandler(StatusChange_DoWork);
            this.StatusChange.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(StatusChanged_ProgressChanged);


            InitializeComponent();
        }

        public void InitImports()
        {
            Boolean is_init = false;

            using (StreamReader reader = new StreamReader("confs\\init.conf"))
            {
                string text = reader.ReadToEnd();
                if (text == "")
                {
                    ProcessStartInfo startInfo = new ProcessStartInfo("cmd");

                    string script1 = "python -m pip install --upgrade pip";
                    string script2 = "pip3 install opencv-python";
                    string script3 = "pip3 install numpy";
                    string script4 = "pip3 install numba";

                    startInfo.UseShellExecute = false;
                    startInfo.CreateNoWindow = false;
                    startInfo.RedirectStandardError = true;
                    startInfo.RedirectStandardInput = true;
                    startInfo.RedirectStandardOutput = false;

                    using (var process = Process.Start(startInfo))
                    {
                        process.StandardInput.WriteLine(script1);
                        process.StandardInput.WriteLine(script2);
                        process.StandardInput.WriteLine(script3);
                        process.StandardInput.WriteLine(script4);
                        process.StandardInput.Close();
                    }
                    is_init = true;
                }
            }
            if (is_init)
            {
                using (StreamWriter writer = new StreamWriter("confs\\init.conf", false))
                {
                    writer.Write("1");
                }
            }
        }

        private void CollectData_DoWork(object sender, DoWorkEventArgs e)
        {
            Console.WriteLine("work");
            ProcessStartInfo startInfo = new ProcessStartInfo("python");

            string dir = System.IO.Directory.GetCurrentDirectory();
            string script = "ScratchDet.py " + " -i \"" + Im + "\"";
            //Console.WriteLine(dir);
            //Console.WriteLine(script);
            startInfo.WorkingDirectory = dir;
            startInfo.Arguments = script;
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = true;
            startInfo.RedirectStandardError = true;
            startInfo.RedirectStandardOutput = true;

            Console.WriteLine("start1");
            using (var process = Process.Start(startInfo))
            {
                Console.WriteLine("start2");
                errors = process.StandardError.ReadToEnd();
                outp = process.StandardOutput.ReadToEnd();
            }
            is_work = false;
            Console.WriteLine("end");
        }

        public delegate void MyDelegate(Label myControl);
        public void DelegateMethod(Label label)
        {
            ProcessButton.IsEnabled = true;
           
            FileInfo f = new FileInfo(Im);
            BitmapImage bi3 = new BitmapImage();
            bi3.BeginInit();
            bi3.UriSource = new Uri(System.IO.Directory.GetCurrentDirectory() + "\\out_img\\" + f.Name, UriKind.Absolute);
            bi3.EndInit();
            AfterIm.Source = bi3;
            if (errors != "" && !errors.Contains("WARN"))
            {
                ErrorLabel.Content = errors;
                Height = 789.2;
            }
            //Im.Split('\\');
            StatusLabel.Foreground = System.Windows.Media.Brushes.Green;
            StatusLabel.Content = "Выполнено";
        }
        private void StatusChange_DoWork(object sender, DoWorkEventArgs e)
        {
            String[] work = { "Обработка", "Обработка.", "Обработка..", "Обработка..." };
            int i = 0;
            while (is_work)
            {
                if (i > 3) i = 0;
                Console.WriteLine("work2");
                StatusChange.ReportProgress(i);
                i++;
                Thread.Sleep(500);
            }
            object[] myArray = new object[1];
            myArray[0] = StatusLabel;
            StatusLabel.Dispatcher.Invoke(new MyDelegate(DelegateMethod), myArray);
        }

        private void StatusChanged_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            int i = e.ProgressPercentage;
            //Console.WriteLine("pr" + i);
            if (i > 3) i = 0;
            String[] work = { "Обработка", "Обработка.", "Обработка..", "Обработка..." };
            StatusLabel.Content = work[i];
            //label3.Refresh();
            Console.WriteLine("loop");

        }


        private void ToolBar_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            DragMove();
        }
        private void MinimizeButton_Click(object sender, RoutedEventArgs e)
        {
            WindowState = WindowState.Minimized;
        }
        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private void ProcessButton_Click(object sender, RoutedEventArgs e)
        {
            //Console.WriteLine("butt");
            StatusChange.ProgressChanged += new ProgressChangedEventHandler(StatusChanged_ProgressChanged);
            if (CollectData.IsBusy != true)
            {
                CollectData.RunWorkerAsync();
            }
            is_work = true;
            ProcessButton.IsEnabled = false;
            if (StatusChange.IsBusy != true)
            {
                StatusLabel.Foreground = System.Windows.Media.Brushes.Red;
                StatusChange.RunWorkerAsync();
            }
            if (StatusChange.WorkerSupportsCancellation == true)
            {
                StatusChange.CancelAsync();
            }
            if (CollectData.WorkerSupportsCancellation == true)
            {
                CollectData.CancelAsync();
            }
            //if (errors.Contains("FileNotFoundError"))
            //    errors = "";

            //Console.WriteLine("tik");
        }

        private void Observe_Click(object sender, RoutedEventArgs e)
        {
            openFileDialog.Filter = "Изображение|*.jpg; *.png; *.bmp";
            if (openFileDialog.ShowDialog().Value)
            {
                UrlTextBox.Text = openFileDialog.FileName;
                Im = openFileDialog.FileName;
                BitmapImage bi3 = new BitmapImage();
                bi3.BeginInit();
                bi3.UriSource = new Uri(Im, UriKind.Absolute);
                bi3.EndInit();
                BeforeIm.Source = bi3;
            }
        }
    }
}
