
@echo off 
setlocal ENABLEDELAYEDEXPANSION
set subjects=2001 2001 2001 2001 2003 2003 2003 2003 2008 2008 2008 2008 2009 2009 2009 2009
set sessions=01N 03N 05N 06N 01N 03N 04N 05N 01N 03N 04N 05N 02N 03N 04N 05N
set dates=20230616 20230620 20230703 20230704 20230718 20230724 20230725 20230731 20231026 20231106 20231107 20231109 20231113 20231114 20231120 20231121

set i=0
(for %%s in (%subjects%) do (
   set subs[!i!]=%%s
   set /A i+=1
))

set i=0
(for %%s in (%sessions%) do (
   set ses[!i!]=%%s
   set /A i+=1
))

set i=0
(for %%d in (%dates%) do (
   set dat[!i!]=%%d
   set /A i+=1
))

(for /L %%x in (0,1,15) do (
   call echo %%subs[%%x]%% %%ses[%%x]%% %%dat[%%x]%%
   call echo python 1_forward_model_auto.py %%subs[%%x]%% %%ses[%%x]%%
   call echo python 2_preprocessing_auto.py %%subs[%%x]%% %%ses[%%x]%% %%dat[%%x]%%
   call echo python 3_extract_source_epochs_auto.py %%subs[%%x]%% %%ses[%%x]%%
))

endlocal
