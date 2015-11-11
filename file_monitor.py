# python script to monitor file changes

import os
import win32file
import win32con
import win32api
import time
import collections
import glob

class savedfiles:
    '''Class to handle file modification time storage and processing'''
    def __init__(self, path):
        self.path = path

    def document_list(self, specify_file = True):
        '''get a list of spricified file (defualt) or all files in the path'''
        # by default look for specific file
        if specify_file:
            # search by filename prefix, more complex criteria must be manullely modify
            prefix = 'Bookmarks'
            file_criteria = prefix + '*'
            file_path = os.path.join(self.path, file_criteria)
            file_list = glob.glob(file_path)

            # sort the file list based on the modification time of the files
            # by default sort give results in a increasing order (old files in the first, new file at the end of the list)
            file_list.sort(key=lambda x: os.path.getmtime(x))
            return file_list
        # if specify_file flag is False, look for all files/directories in the path
        else:
            dir_criteria = '*'
            dir_path = os.path.join(self.path, dir_criteria)
            dir_list = glob.glob(dir_path)
            dir_list.sort(key=lambda x: os.path.getmtime(x))
            return dir_list

    def queue_all(self, maxnum):
        '''Generate a deque data structure with list of all documents in the path.
        Maxnum is the maxnum length of the deque.
        By default deque drop the element from opposite side of the adding point
        when the number of the element exceeds maxnum.
        First time define a deque is equivalent as append a list, therefore droping the exceeding elements from left side.'''
        alldoclist = self.document_list(specify_file=False)
        queue = collections.deque(alldoclist, maxlen = maxnum)
        return queue

    def queue_file(self, maxnum):
        '''Gernerate a deque data structure with the list of specific files in the path.'''
        filelist = self.document_list()
        queue = collections.deque(filelist, maxlen = maxnum)
        return queue

path_to_watch = r"C:\Users\username\AppData\Local\Google\Chrome\User Data\Default" # directory to monitor
file_to_watch = 'Bookmarks' # the file to monitor
path_to_move= r"D:\Google Drive\backup\bookmarks" # the directory to copy the changed file
full_file_path = os.path.join(path_to_watch, file_to_watch)

# use a deque data struture to store modification time footprint
numfile_to_keep = 5
backupfiles = savedfiles(path_to_move)
filequeue = backupfiles.queue_file(numfile_to_keep)
print(filequeue)
#path_to_watch = filename
#os.path.abspath(filename)
#path_to_storechange = destfile
#os.path.abspath(destfile)


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

def timetoname(path, filename, intime):
    '''function to convert path, filename and input time to new file name
    the variable intime can not be named as 'time', because time is here a class with gmtime method'''
    timestr = time.strftime("%Y%m%d_%H%M%S", time.gmtime(intime))
    newfilename = '_'.join((filename, timestr))
    newfilepath = os.path.join(path, newfilename)
    return newfilepath

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
        # add time delay to wait until the tmp file become real file
        time.sleep(0.1)

        # get modify time stamp and use it in saved filename
        try:
            mtime = os.path.getmtime(full_file_path)
        except OSError:
            mtime = 0

        if len(filequeue) >= 5:
            filetoremove = filequeue.popleft()
            os.remove(filetoremove)

        # backup newfiles generated only longer than 1 second after the old file
        mtime_lastfile = os.path.getmtime(filequeue[-1])
        if mtime - mtime_lastfile > 1:
            full_copy_path = timetoname(path_to_move, file_to_watch, mtime)
            # copy the file
            win32api.CopyFile(full_file_path, full_copy_path)
            # append the new modification time to the right side of the deque
            filequeue.append(full_copy_path)
