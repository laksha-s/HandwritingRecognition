from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download