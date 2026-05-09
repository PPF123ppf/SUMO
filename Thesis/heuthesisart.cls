%%
%% This is file `examples/art/reports/heuthesisart.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% heuthesis.dtx  (with options: `artcls')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2021-2026 by LIWenhui <lwh@hrbeu.edu.cn>
%% 
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3a
%% of this license or (at your option) any later version.
%% The latest version of this license is in:
%% 
%% http://www.latex-project.org/lppl.txt
%% 
%% To produce the documentation run the original source files ending with `.dtx'
%% through LaTeX.
%% 
\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{heuthesisart}
[2021/05/06 3.0.15 Harbin Engineering University Thesis Template]
\RequirePackage{ifthen}
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
  family=heu,
  prefix=heu@,
  setkeys=\kvsetkeys}

\newif\ifheu@bachelor
\newif\ifheu@master
\newif\ifheu@doctor
\define@key{heu}{type}{%
  \heu@bachelorfalse
  \heu@masterfalse
  \heu@doctorfalse
  \expandafter\csname heu@#1true\endcsname}

\newif\ifheu@productdevelopment
\newif\ifheu@projectdplanning
\newif\ifheu@projectdesign
\newif\ifheu@applicationresearch
\define@key{heu}{researchtype}{%
  \heu@productdevelopmentfalse
  \heu@projectdplanningfalse
  \heu@projectdesignfalse
  \heu@applicationresearchfalse
  \expandafter\csname heu@#1true\endcsname}
\ifheu@productdevelopment\relax\else
\ifheu@projectdplanning\relax\else
\ifheu@projectdesign\relax\else
\heu@applicationresearchtrue
\fi
\fi
\fi

\newif\ifheu@qingdao
\newif\ifheu@yantai
\newif\ifheu@harbin
\define@key{heu}{campus}{%
  \heu@qingdaofalse
  \heu@yantaifalse
  \heu@harbinfalse
  \expandafter\csname heu@#1true\endcsname}
\ifheu@yantai\relax\else
\ifheu@qingdao\relax\else
\heu@harbintrue
\fi
\fi

\newif\ifheu@opening
\newif\ifheu@midterm
\define@key{heu}{stage}{%
  \heu@openingfalse
  \heu@midtermfalse
  \expandafter\csname heu@#1true\endcsname}

\DeclareBoolOption[true]{raggedbottom}
\DeclareBoolOption[false]{pifootnote}
\DeclareBoolOption[false]{debug}
\DeclareBoolOption[true]{toc}
\DeclareBoolOption[true]{newtxmath}

\DeclareStringOption{fontset}

\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{ctexart}}

\ProcessKeyvalOptions*
\PassOptionsToPackage{no-math}{fontspec}

\ifheu@bachelor\relax\else
\ifheu@master\relax\else
\ifheu@doctor\relax\else
\ClassError{heuthesisart}%
{
  \MessageBreak Please specify thesis type in option:
  \MessageBreak type=[bachelor | master | doctor]
}
\fi
\fi
\fi

\ifheu@productdevelopment\relax\else
\ifheu@projectdplanning\relax\else
\ifheu@projectdesign\relax\else
\ifheu@applicationresearch\relax\else
\ClassError{heuthesisartplus}%
{
  \MessageBreak Please specify researchtype in option:
  \MessageBreak researchtype=[productdevelopment | projectdplanning | projectdesign | applicationresearch]
}
\fi
\fi
\fi
\fi

\ifheu@opening\relax\else
\ifheu@midterm\relax\else
\ClassError{heuthesisart}%
{
  \MessageBreak Please specify stage in option:
  \MessageBreak stage=<opening|midterm>
}
\fi
\fi

\ifthenelse%
{\equal{\heu@fontset}{}}%
{%
  \PassOptionsToPackage{AutoFakeBold=2}{xeCJK}
}%
{%
  \ifthenelse%
  {\equal{\heu@fontset}{siyuan}}%
  {\relax}%
  {%
    \PassOptionsToPackage{AutoFakeBold=2}{xeCJK}
  }%
  \PassOptionsToClass{fontset=\heu@fontset}{ctexart}
}%
\LoadClass[a4paper,UTF8,zihao=-4,scheme=chinese]{ctexart}

\RequirePackage{etoolbox}
\RequirePackage{ifxetex}
\ifxetex
\else
  \ClassError{heuthesis}%
     {Please use: \MessageBreak
     xelatex}{}
\fi
\RequirePackage{xparse}
\RequirePackage{amsmath}
\RequirePackage[amsmath,thmmarks,hyperref]{ntheorem}
\RequirePackage{amssymb}
\ifheu@newtxmath
\RequirePackage{newtxmath}
\fi
\RequirePackage{graphicx}
\RequirePackage{pdfpages}
\includepdfset{fitpaper=true}
\RequirePackage{enumitem}       %使用enumitem宏包,改变列表项的格式
\RequirePackage{environ}
\ifheu@raggedbottom
  \RequirePackage[bottom,perpage,hang]{footmisc}
  \raggedbottom
\else
  \RequirePackage[perpage,hang]{footmisc}
\fi
\RequirePackage{bbding}
\ifheu@pifootnote
  \RequirePackage{pifont}
\fi
\RequirePackage{xeCJKfntef}
\RequirePackage{diagbox}
\RequirePackage{longtable}
\RequirePackage{booktabs}
\RequirePackage[sort&compress,numbers]{natbib}
\RequirePackage{hyperref}
\hypersetup{%
  CJKbookmarks=true,
  linktoc=all,
  bookmarksnumbered=true,
  bookmarksopen=true,
  bookmarksopenlevel=1,
  breaklinks=true,
  colorlinks=false,
  plainpages=false,
  pdfborder=0 0 0}
\urlstyle{same}

\RequirePackage{geometry}
\geometry{
  a4paper,
  hcentering,
  ignoreall,
  nomarginpar,
  centering,
  text={150true mm,236true mm},
  left=30true mm,
  head=5true mm,
  headsep=2true mm,
  footskip=0true mm,
  foot=5.2true mm
}

\ifheu@debug%
\RequirePackage{layout}
\RequirePackage{layouts}
\RequirePackage{lineno}
\fi

\RequirePackage{fancyhdr}
\RequirePackage{tabularx}
\RequirePackage{varwidth}
\RequirePackage{changepage}
\RequirePackage{multicol}
\RequirePackage[below]{placeins}% 允许上一个section的浮动图形出现在下一个section的开始部分,还提供\FloatBarrier命令,使所有未处理的浮动图形立即被处理
\RequirePackage{flafter}        % 使得所有浮动体不能被放置在其浮动环境之前，以免浮动体在引述它的文本之前出现.
\RequirePackage{multirow}       % 使用Multirow宏包，使得表格可以合并多个row格
\RequirePackage{subfigure}      % 支持子图 %centerlast 设置最后一行是否居中
\RequirePackage[subfigure]{ccaption} % 支持双语标题
\RequirePackage{xltxtra}

\renewcommand\normalsize{%
  \@setfontsize\normalsize{12bp}{20.50398bp}%
  \abovedisplayskip=8pt
  \abovedisplayshortskip=8pt
  \belowdisplayskip=\abovedisplayskip
  \belowdisplayshortskip=\abovedisplayshortskip}

\def\heu@def@fontsize#1#2{%
  \expandafter\newcommand\csname #1\endcsname[1][1.3]{%
    \fontsize{#2}{##1\dimexpr #2}\selectfont}}
\heu@def@fontsize{dachu}{58.0bp}
\heu@def@fontsize{chuhao}{42.2bp}
\heu@def@fontsize{xiaochu}{36.1bp}
\heu@def@fontsize{yihao}{26.1bp}
\heu@def@fontsize{xiaoyi}{24.1bp}
\heu@def@fontsize{erhao}{22.1bp}
\heu@def@fontsize{xiaoer}{18.1bp}
\heu@def@fontsize{sanhao}{16.1bp}
\heu@def@fontsize{xiaosan}{15.1bp}
\heu@def@fontsize{sihao}{14.1bp}
\heu@def@fontsize{banxiaosi}{13.0bp}
\heu@def@fontsize{xiaosi}{12.1bp}
\heu@def@fontsize{dawu}{11.0bp}
\heu@def@fontsize{wuhao}{10.5bp}
\heu@def@fontsize{xiaowu}{9.0bp}
\heu@def@fontsize{liuhao}{7.5bp}
\heu@def@fontsize{xiaoliu}{6.5bp}
\heu@def@fontsize{qihao}{5.5bp}
\heu@def@fontsize{bahao}{5.0bp}

\ctexset{%
  section={
    afterindent=true,
    beforeskip={7.5bp},%上下空0.5行
    afterskip={7.5bp},
    format={\heiti\xiaosan[1.25]},
    aftername=\enspace,
    fixskip=true,
    break={},
  },
  subsection={
    afterindent=true,
    beforeskip={7bp},
    afterskip={7bp},
    format={\heiti\sihao[1.25]},
    aftername=\enspace,
    fixskip=true,
    break={},
  },
  subsubsection={
    afterindent=true,
    beforeskip={3bp},
    afterskip={3bp},
    format={\heiti\normalsize},
    aftername=\enspace,
    fixskip=true,
    break={},
  },
  paragraph/afterindent=true,
  subparagraph/afterindent=true
}

\def\heu@def@term#1{%
  \define@key{heu}{#1}{\csname #1\endcsname{##1}}
  \expandafter\gdef\csname #1\endcsname##1{%
    \expandafter\gdef\csname heu@#1\endcsname{##1}}
  \csname #1\endcsname{}}

\heu@def@term{ctitlecover} %中文标题封面
\heu@def@term{csubject}
\heu@def@term{cauthor}
\heu@def@term{cstudentid}
\heu@def@term{cclassid}
\heu@def@term{caffil}
\heu@def@term{csupervisor}
\heu@def@term{csubmitdate}
\heu@def@term{coralexdate}
\heu@def@term{cdate}

\def\heu@parse@keywords#1{
  \define@key{heu}{#1}{\csname #1\endcsname{##1}}
  \expandafter\gdef\csname heu@#1\endcsname{}
  \expandafter\gdef\csname #1\endcsname##1{
    \@for\reserved@a:=##1\do{
      \expandafter\ifx\csname heu@#1\endcsname\@empty\else
      \expandafter\g@addto@macro\csname heu@#1\endcsname{%
        \ignorespaces\csname heu@#1@separator\endcsname}
      \fi
      \expandafter\expandafter\expandafter\g@addto@macro%
      \expandafter\csname heu@#1\expandafter\endcsname\expandafter{\reserved@a}}}}

\def\heusetup{\kvsetkeys{heu}}

\newcommand{\heu@report@titlepage@graduate}{
  \ifthenelse%
  {\equal{\heu@fontset}{siyuan}}%
  {\xiaosi[1]\vspace*{0.65em}}%
  {\xiaosi[1]\textcolor[rgb]{1,1,1}{\songti{\heu@motto}}}%
  \vspace*{10mm}
  \begin{center}
    \kaishu\xiaoer\textbf{\heu@cschoolname\ifheu@qingdao\heu@qingdaocampus\fi\ifheu@yantai\heu@yantaicampus\fi}
  \end{center}
  \vspace{1mm}
  \begin{center}
    \songti\erhao\textbf{\heu@cxuewei\heu@cthesisname
      \ifheu@opening
      \heu@stage@opening
      \else
      \ifheu@midterm
      \heu@stage@midterm
      \fi
      \fi
      \heu@stage@doctype
    }
  \end{center}
   \vspace{10mm}
   \parbox[t][3cm][t]{\textwidth}{
     \begin{center}
       \songti\xiaoer\textbf{\heu@cthesistitleprefix\heu@title@csep\heu@ctitlecover}
     \end{center}
   }
     \parbox[b][3cm][t]{\textwidth}{
       \begin{center}\songti\sanhao
       \renewcommand{\arraystretch}{1.8}
       \begin{tabular}{l@{\ \ }c}
         \textbf{\heu@graduate@caffiltitle：} & \underline{\makebox[6.1cm]{\textbf{\heu@caffil}}}\\
         \textbf{\heu@graduate@cmajortitle：} & \underline{\makebox[6.1cm]{\textbf{\heu@csubject}}}\\
         \textbf{\heu@graduate@supervisor：} & \underline{\makebox[6.1cm]{\textbf{\heu@csupervisor}}}\\
         \textbf{\heu@graduate@studenttitle：} & \underline{\makebox[6.1cm]{\textbf{\heu@cauthor}}}\\
         \textbf{\heu@graduate@studentid：} & \underline{\makebox[6.1cm]{\textbf{\heu@cstudentid}}}\\
         \textbf{\heu@graduate@datetitle：} & \underline{\makebox[6.1cm]{\textbf{\heu@cdate}}}\\
       \end{tabular}\renewcommand{\arraystretch}{1}
     \end{center}
     }
   \vfill
   \ifheu@harbin
     \heu@harbin@schoolbottommark
   \else
   \ifheu@qingdao
     \heu@qingdao@schoolbottommark
   \else
   \ifheu@yantai
     \heu@yantai@schoolbottommark
   \fi
   \fi
}

\newcommand{\heu@report@titlepage@bachelor}{
   \ifthenelse%
   {\equal{\heu@fontset}{siyuan}}%
   {\xiaosi[1]\vspace*{0.65em}}%
   {\xiaosi[1]\textcolor[rgb]{1,1,1}{\songti{\heu@motto}}}%
   \vspace*{10mm}
   \begin{center}
     \includegraphics[width=6.2cm]{heulogo}
   \end{center}
   \begin{center}
     \songti\xiaoyi\textbf{\heu@bachelor@cthesisname
       \ifheu@opening
         \heu@stage@opening
       \else
         \ifheu@midterm
           \heu@stage@midterm
         \fi
       \fi
       \heu@stage@doctype
     }
   \end{center}
   \vspace{15mm}
   \parbox[t][4.0cm][t]{\textwidth}{
     \begin{center}
       \songti\xiaoer\textbf{\heu@cthesistitleprefix\heu@title@csep\heu@ctitlecover}
     \end{center}
   }
     \parbox[b][6cm][t]{\textwidth}{
       \begin{center}\songti\sanhao
       \renewcommand{\arraystretch}{2.1}
       \begin{tabular}{l@{\ \ }c}
         \textbf{\heu@bachelor@cmajortitle} & \underline{\makebox[6.1cm]{\textbf{\heu@csubject}}}\\
         \textbf{\heu@bachelor@cstudenttitle} & \underline{\makebox[6.1cm]{\textbf{\heu@cauthor}}}\\
         \textbf{\heu@bachelor@cstudentidtitle} & \underline{\makebox[6.1cm]{\textbf{\heu@cstudentid}}}\\
         \textbf{\heu@bachelor@csupervisortitle} & \underline{\makebox[6.1cm]{\textbf{\heu@csupervisor}}}\\
         \textbf{\heu@bachelor@cdatetitle} & \underline{\makebox[6.1cm]{\textbf{\heu@cdate}}}\\
       \end{tabular}\renewcommand{\arraystretch}{1}
     \end{center}
     }
   \vfill
   \ifheu@yantai
     \relax
   \else
     \heu@harbin@bachelor@schoolbottommark
   \fi
}

\newcommand{\heu@report@backpage@bachelor}{
  \thispagestyle{empty}
  \noindent\parbox[t][6.5cm][t]{\textwidth}{\heu@bachelor@teachercomment}
  \noindent\parbox[b][6cm][t]{\textwidth}{\heu@bachelor@teachersign\underline{\makebox[3cm]{}}\hfill\heu@bachelor@checkdate\underline{\makebox[3cm]{}}}
}

\renewcommand\tableofcontents{%
  \thispagestyle{empty}
  \normalsize\@starttoc{toc}
}

\def\makecover{
  \begin{titlepage}
    \ifheu@bachelor
      \heu@report@titlepage@bachelor
    \else
      \heu@report@titlepage@graduate
    \fi
    \clearpage
    \ifheu@toc
      \tableofcontents
      \clearpage
    \fi
  \end{titlepage}
}
\def\makebackcover{
    \clearpage
    \heu@report@backpage@bachelor
}

\newcommand\bibstyle@numerical{\bibpunct{[}{]}{,}{s}{,}{\textsuperscript{,}}}
\newcommand\bibstyle@inline{\bibpunct{[}{]}{,}{n}{,}{\heu@inline@sep}}

\citestyle{numerical}
\DeclareRobustCommand\inlinecite{\@inlinecite}
\def\@inlinecite#1{\begingroup\citestyle{inline}\let\@cite\NAT@citenum\citep{#1}\endgroup}
\let\onlinecite\inlinecite
\newcommand{\lcite}{\inlinecite}
\renewenvironment{thebibliography}[1]{%
  \list{\@biblabel{\@arabic\c@enumiv}}%
  {\renewcommand{\makelabel}[1]{##1\hfill}
    \settowidth{\labelwidth}{\@biblabel{#1}}
    \setlength{\labelsep}{0.5em}
    \setlength{\itemindent}{0pt}
    \setlength{\leftmargin}{\labelsep+\labelwidth}
    \addtolength{\itemsep}{-0.8em}
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
  \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy\frenchspacing
  \flushbottom
  \clubpenalty0
  \@clubpenalty \clubpenalty
  \widowpenalty0%
  \interlinepenalty-50%
\sfcode`\.\@m}
{\def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
\endlist\frenchspacing}
\AtEndOfClass{\input{heuthesisart.cfg}}
\AtEndOfClass{\sloppy}
\endinput
%%
%% End of file `examples/art/reports/heuthesisart.cls'.
