# python script to monitor file changes

import os
import win32file
import win32con
import win32api
import time

path_to_watch = r"C:\Users\username\AppData\Local\Google\Chrome\User Data\Default" # directory to monitor
file_to_watch = 'Bookmarks' # the file to monitor
path_to_move= r"D:\Google Drive\backup\bookmarks\Bookmarks_bk" # the directory to copy the changed file

FILE_LIST_DIRECTORY = 0x0001
hDir = win32file.CreateFile(
        path_to_watch,
        FILE_LIST_DIRECTORY,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE,
        None,
        win32con.OPEN_EXISTING,
        win32con.FILE_FLAG_BACKUP_SEMANTICS,
        None
        )

while True:
    results = win32file.ReadDirectoryChangesW (
            hDir,
            1024,
            False,
            win32con.FILE_NOTIFY_CHANGE_SIZE |
            win32con.FILE_NOTIFY_CHANGE_LAST_WRITE,
            None,
            None
            )
    changed_file = results[0][1]
    if file_to_watch in changed_file:
        full_file_path = os.path.join(path_to_watch, file_to_watch)
        time.sleep(0.1) # add time delay to wait until the tmp file become real file
        win32api.CopyFile(full_file_path, path_to_move)
