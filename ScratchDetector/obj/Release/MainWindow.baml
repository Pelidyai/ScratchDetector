   M S B A M L   `   `   `  ���� J  FScratchDetector, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null1clr-namespace:ScratchDetectorScratchDetector       ScratchDetector.MainWindow   \ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     @d2http://schemas.microsoft.com/expression/blend/2008     Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     /localclr-namespace:ScratchDetector       � Title$  Chatter��$��589.2��$	��838��
������  � AllowsTransparency .  � WindowStyle$
 None=� � 	Resources    WindowColor  ContentAreaColorLight  SelectedBackgroundColor  SelectedUnfocusedColor  ControlLightColor  ControlMediumColor  ControlPressedColor  
GlyphColor  DisabledBorderLightColor 	 TextBoxTemplate �� 
TargetType �� CornerRadius ��Name �� 
TargetType �� Triggers	 P� Property
 P� Value �� Property �� Value �� Triggers :� RoutedEvent g� Duration �� TargetProperty	 W� To 
 TextBoxBase �� BasedOn  TextBoxEvented��%I  &	        &	      &	 &     &	 9     &	 L     &	 _     &	 r     &	 �     &	 �     &		 �     '�� '    '�� i    &	
 �    7��7 	 &	 �    7 
 ��	#FF40444B����	#FFC5CBF9����	#FF40444B����	#FFDDDDDD����	#9940444B����	#4440444B����	#5540444B����	#CC888EFF����	#FFAAAAAA���� " ��.R��� $��Border��$ 5��$��1q�#��z��._��� $ PART_ContentHost������ " �� P� 	 �@J�IsMouseOver
 . . ���  � ��
 �������  �@�Cursor$ IBeam�� :� $ Mouse.MouseEnter��[��� .���� .��W� $ 0:0:0.2D�$ FontSize�$ 16X�:� $ Mouse.MouseLeave��[��� .���� .��W� $ 0:0:0.2D�$ FontSize�$ 15X��� " ��.���  � �� �� " ��.R��� $ 5��$��3q����	#FF36393F#��z������ " �� P� 	 �@J�IsMouseOver
 . . ���  � �� �	#9936393F._��� #��z��#��z�� P� 	 �@J�IsMouseOver
 . . ���  � �� �	#FF40444B��  � �� �	#CC888EFF��  � �� �	#9936393F�� " ��.���  � �� �� " ��.R��� #��z��#��z����  � �� �	#CC888EFF�� " ��8   .���  � �� �	#9940444B��  � ��$ 15(���  �@��AcceptsReturn . ��  � ��8  ��  � ��
 ����� P� 	 �@J�	IsFocused
 . . ���  � �� �	#FF40444B�� 8   " ��.�-   .���� $ 	40,0,40,0��$��0q����	#FF202225$��	0,0,0.4,0q�._�� -   $��	InnerGrid��$��Stretch=�$��Stretch=�$��10,40,10,10q�.+��� $ 30�����	#FF2F3136$��
0,0,0,-0.4q��� -   $��isCont�����	#CC888EFF$8��2Отобразить контуры лопатки��$
��Left=�$��21��$��34,5,0,0q�$	��Top=�$	��187���� -   $��Observe��$��
Обзор��$
��Left=�$��27��$��586.8,443,0,0q�$	��Top=�$	��205���� -   $��	ChanelUrl��$��Изображение��$
��Left=�$��10,422.4,0,0q�$	��Top=� J� RenderTransformOrigin$ 2.469,-2.984&��� -   $��
UrlTextBox��#��[ $
��Left=�$��27��$��
10,443,0,0q� �� TextWrapping$
 Wrap=�$?��9Абсолютный путь до изображения��$	��Top=�$��571.8��
�������$��15(��� -   $��ProcessButton��$��Обработать��$
��Left=�$��27��$��
10,475,0,0q�$	��Top=�$��781.8���� -   $��StatusLabel��$�� ��$
��Left=�$��26��$��	619,0,0,0q�$	��Top=�$	��173���� -	   $��
ErrorLabel��$�� ��$
��Left=�$��10,507,0,-166.2q�$	��Top=�$	��782��$	��198��� $��10,31,10.4,113.8q�.+��� -
   $��AfterIm��$-��'pack://application:,,,/Images/cross.png��$��	384,0,0,0q��� -   $��BeforeIm��$-��'pack://application:,,,/Images/cross.png��$��	0,0,384,0q�� -   $��ToolBar�����	#FF202225$��Stretch=�$��35��$��34,-40,75,0q�$	��Top=�� $
��Left=�$��35��$��743.4,-40,0,0q�$	��Top=�$��75��.+��� -   $��CloseButton��$
��Left=�$��32��$��40,3,0,0q�$	��Top=�$��35��.���� $-��'pack://application:,,,/Images/cross.png���� -   $��MinimizeButton��$
��Left=�$��32��$��0,3,0,0q�$	��Top=�$��35��.���� $,��&pack://application:,,,/Images/line.png��