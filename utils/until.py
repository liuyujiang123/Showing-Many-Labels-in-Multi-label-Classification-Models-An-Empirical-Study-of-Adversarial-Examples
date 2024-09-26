#@Time      :2021/1/22 14:13
#@Author    :zhounan
#@FileName  :until.py
import os
import logging

def new_folder(file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def init_log(log_file):
  new_folder(log_file)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

  fh = logging.FileHandler(log_file)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)

  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)

def write_to_excel(metrics, base_row, args, sheet, filename):
    import xlwt
    import xlrd
    from xlutils.copy import copy
    rb = xlrd.open_workbook('../attack_result/{}'.format(filename))
    wb = copy(rb)

    if args.adv_method == 'ml_cw':
        row = 0 + base_row
    elif args.adv_method == 'ml_deepfool':
        row = 1 + base_row
    elif args.adv_method == 'FGSM':
        row = 2 + base_row
    elif args.adv_method == 'MI-FGSM':
        row = 3 + base_row
    elif args.adv_method == 'mla_lp':
        row = 4 + base_row

    wb.get_sheet(sheet).write(row, 3, '{:.4f}'.format(metrics['attack rate']))
    wb.get_sheet(sheet).write(row, 4, '{:.4f}'.format(metrics['norm']))
    wb.get_sheet(sheet).write(row, 5, '{:.4f}'.format(metrics['norm_1']))
    wb.get_sheet(sheet).write(row, 6, '{:.4f}'.format(metrics['max_r']))
    wb.get_sheet(sheet).write(row, 7, '{:.4f}'.format(metrics['mean_r']))
    wb.get_sheet(sheet).write(row, 8, '{:.4f}'.format(metrics['rmsd']))

    wb.save('../attack_result/{}'.format(filename))