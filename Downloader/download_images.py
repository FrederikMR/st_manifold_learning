#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:56:43 2023

@author: fmry
"""

#%% Sources

#https://www.geeksforgeeks.org/how-to-download-all-images-from-a-web-page-in-python/

#%% Modules

from Downloader.setup import *

#%% Download one page

def download_images(images, folder_name, base_url='', page_num=0):
   
    # initial count is zero
    count = 0
    
    N_images = len(images)
    if N_images != 0:
        for i, image in enumerate(images):
            try:
                image_link = image["src"]
                r = requests.get(''.join((base_url, image_link))).content
                with open(f"{folder_name}/images{i+1+page_num}.jpg", "wb+") as f:
                    f.write(r)
                count += 1
            except:
                pass
           
    print(f"-Downloaded {count}/{N_images}")
    
    return count
    
#%% Iterater

def iterator(folder_name='', base_url='https://orbit.dtu.dk'):
    
    #for i in range(155):
    i = 2
    url = 'https://orbit.dtu.dk/en/persons/'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    N_pages = int(soup.findAll('a', {'class': 'step'})[-1]['href'].split("page=",1)[1])+1
    
    page_num = 0
    for i in range(N_pages):
        print(f"Iteration {i+1}/{N_pages-1}")
        
        url = f'https://orbit.dtu.dk/en/persons/?format=&page={i}'

        r = requests.get(url)
         
        # Parse HTML Code
        soup = BeautifulSoup(r.text, 'html.parser')
         
        # find all images in URL
        images = soup.findAll('img')

        page_num += download_images(images[1:-1], folder_name, base_url, page_num=page_num)
        
def clean_data(folder_name='dtu_images', remove_folder='remove_images'):
    
    images = os.listdir(folder_name)
    removes = os.listdir(remove_folder)
    
    N_images = len(images)
    N_removes = len(removes)
    
    for i in range(N_images):
        img = os.path.join(folder_name, images[i])
        for j in range(N_removes):
            rem = os.path.join(remove_folder, removes[j])
            if open(img,"rb").read() == open(rem,"rb").read():
                os.remove(img)
                break
            
    N_new = len(os.listdir(folder_name))
    
    print(f"Remaining images are {N_new}/{N_images}")
    
    return

def rename_data(folder_name='data/dtu_images'):
    
    images = os.listdir(folder_name)
    
    N_images = len(images)
    
    for i in range(N_images):
        img = os.path.join(folder_name, images[i])
        os.rename(img, os.path.join(folder_name, f'images_{i+1}.jpg'))
    
    return