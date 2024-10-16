# How to Make Virtul Environment
# Choose Project Layout
# Setup Metadata
# Setup Documentation Builder
# Version Control on Git and Github
# How to Start Local Development
# Release The Package to PyPi or Conda
# How to start virtual environment

If you are using PowerShell windows, you might get error like this
```powershell
C:\Users\{your-user}\{your-package}>\env\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on 
this system. For more information, see about_Execution_Policies at https:/go.microsoft.com/fwlink/?LinkID=135170.
At line:1 char:1
+ .\env\Scripts\activate.ps1
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
```
if so, you can fix the error by using the following command
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```
finally, you can use usual comman to activate your virtual environment
```powershell
C:\Users\{your-user}\{your-package}>\env\Scripts\Activate.ps1
```