# python script to monitor file changes and backup files

import os
import win32file
import win32con
import win32api
import time
import collections
import glob
import shutil

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


def timetoname(path, filename, intime, isdst=False):
    '''function to convert path, filename and input time to new file name
    the variable intime can not be named as 'time', because time is here a class with gmtime method
    Isdst need to be explicitly defined, because time.localtime() always return is_dst=0 on this Windows machine.'''
    if not isdst:
        timedelay = 3600
    else:
        timedelay = 0
    timestr = time.strftime("%Y%m%d_%H%M%S", time.gmtime(intime+timedelay))
    newfilename = '_'.join((filename, timestr))
    newfilepath = os.path.join(path, newfilename)
    return newfilepath

def main():
    path_to_watch = r"C:\Users\username\AppData\Local\Google\Chrome\User Data\Default" # directory to monitor
    file_to_watch = 'Bookmarks' # the file to monitor
    path_to_move= r"D:\Google Drive\backup\bookmarks" # the directory to copy the changed file
    full_file_path = os.path.join(path_to_watch, file_to_watch)

    # use a deque data struture to store modification time footprint
    numfile_to_keep = 5
    backupfiles = savedfiles(path_to_move)
    filequeue = backupfiles.queue_file(numfile_to_keep)

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

    # infinitive loop to start monitor
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

            if len(filequeue) >= numfile_to_keep:
                filetoremove = filequeue.popleft()
                # try to remove the oldest file in the queue
                # this can be extanded to a log file generator in the future
                try:
                    os.remove(filetoremove)
                # if file not find, regenerate the filequeue with the updated files
                except FileNotFoundError:
                    newbackupfiles = savedfiles(path_to_move)
                    filequeue = newbackupfiles.queue_file(numfile_to_keep)
                # try to remove a dir will generate PermsssionError
                # use shutil to remove the dir
                # if it is other reason, ignore it
                except PermissionError:
                    shutil.rmtree(filetoremove, ignore_errors=True)

            # backup newfiles generated only longer than 1 second after the old file
            mtime_lastfile = os.path.getmtime(filequeue[-1])
            if mtime - mtime_lastfile > 1:
                full_copy_path = timetoname(path_to_move, file_to_watch, mtime)
                # copy the file
                # win32api.CopyFile(full_file_path, full_copy_path)
                # use shutil instead of win32api for portable reason
                # in Unix shutil.copy2 will copy metadata
                # shutil.copy2 will replace the existed file
                # to avoid overwrite existed file, use shutil.copyfile
                # but then the metadata can not be copied
                try:
                    shutil.copy2(full_file_path, full_copy_path)
                except IOError:
                    pass
                # append the new modification time to the right side of the deque
                filequeue.append(full_copy_path)

if __name__ == "__main__":
    # execute only if run as a script
    main()
