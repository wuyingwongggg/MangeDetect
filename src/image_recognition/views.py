from operator import attrgetter
from django.shortcuts import redirect, render, get_object_or_404
from django.http import JsonResponse
from image_recognition.models import UploadedImage
from image_recognition.forms import UploadImageForm, UpdateUploadImageForm
from account.models import Account
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torchvision.models import ResNet18_Weights
from django.core.exceptions import PermissionDenied
from django.core.cache import cache
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
from django.views import View
from django.http import HttpResponse
import json
from django.core.files.uploadedfile import InMemoryUploadedFile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import io
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render
from datetime import datetime, timedelta
from django.utils import timezone
from django.contrib import messages
from django.urls import reverse
import urllib.parse

# Function for setting rate limits.
def rate_limit(key_prefix, limit, period):
    def decorator(fn):
        def wrapper(request, *args, **kwargs):
            key = f"{key_prefix}:{request.user.id if request.user.is_authenticated else request.META['REMOTE_ADDR']}"
            count = cache.get(key, 0)
            
            if count >= limit:
                raise PermissionDenied("Rate limit exceeded. Please try again later.")
            
            cache.set(key, count + 1, period)
            return fn(request, *args, **kwargs)
        return wrapper
    return decorator

# Define a class mapping for wombat categories.
class_mapping = {
    0: "Normal",
    1: "Mange"
}

# Load and configure the ResNet101 model with dropout.
def load_model():
    model = models.resnet101(weights=None)
    
    # Modify the final fully connected layer with dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_mapping))
    )
    
    # Load custom weights
    try:
        state_dict = torch.load('mange_classification_model_v2.pth', map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
        print("Custom model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading custom weights: {e}")
        raise RuntimeError("Failed to load model weights")
    
    model.eval()  # Set to evaluation mode
    return model

# Initialize transformations for images.
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global variable to store the model
try:
    image_model = load_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    image_model = None

# Make predicton on a single image.
# Returns (predicted_class, confidence_score)
def predict_image(image: Image.Image) -> Tuple[str, float]:
    if image_model is None:
        return "Model not loaded", 0.0
    
    # Prepare image
    try:
        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transformations
        input_tensor = transform_pipeline(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = image_model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Get prediction and confidence
        confidence, prediction = torch.max(probabilities, 0)
        predicted_class = class_mapping[prediction.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error processing image", 0.0

# Process a single image into the model.
def process_single_image(image: Image.Image) -> Tuple[str, float]:
    try:
        prediction, confidence = predict_image(image)
        return prediction, confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error", 0.0

# Process a single uploaded image, and return the image and its result.
def process_uploaded_file(file: InMemoryUploadedFile) -> Tuple[Image.Image, str, float]:
    try:
        image = Image.open(file)
        prediction, confidence = process_single_image(image)
        return image, prediction, confidence
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, "Error", 0.0

@rate_limit('image_upload', 300, 3600)  # Increased rate limit for multiple uploads
def upload_image_view(request):
    if not request.user.is_authenticated:
        return redirect('must_authenticate')
    
    if request.method == 'POST':
        files = request.FILES.getlist('images')
        
        if not files:
            return JsonResponse({'error': 'No files uploaded'}, status=400)
        
        results = []
        author = Account.objects.filter(email=request.user.email).first()
        
        # Process images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
            future_to_file = {executor.submit(process_uploaded_file, file): file 
                            for file in files}
            
            for future in future_to_file:
                file = future_to_file[future]
                try:
                    image, prediction, probability = future.result()
                    
                    # Create UploadedImage instance
                    uploaded_image = UploadedImage(
                        author=author,
                        title=f"Upload {file.name}",
                        image=file,
                        result=f"{prediction} (Confidence: {probability:.2f})"
                    )
                    uploaded_image.save()
                    
                    results.append({
                        'filename': file.name,
                        'prediction': prediction,
                        'probability': probability,
                        'slug': uploaded_image.slug
                    })
                    
                except Exception as e:
                    results.append({
                        'filename': file.name,
                        'error': str(e)
                    })
        
        return JsonResponse({'results': results})
    
    return render(request, "image_recognition/upload.html", {})

# Update the detail view to handle multiple images
def detail_image_view(request, slug):
    context = {}
    uploaded_image = get_object_or_404(UploadedImage, slug=slug)
    
    # Get related images uploaded in the same batch
    related_images = UploadedImage.objects.filter(
        author=uploaded_image.author,
        date_updated__date=uploaded_image.date_updated.date()
    ).exclude(id=uploaded_image.id)[:5]

    context['uploaded_image'] = uploaded_image
    context['related_images'] = related_images

    return render(request, 'image_recognition/detail_image.html', context)

def edit_image_view(request, slug):
    context = {}
    user = request.user
    if not user.is_authenticated:
        return redirect("must_authenticate")
    
    uploaded_image = get_object_or_404(UploadedImage, slug=slug)

    if request.POST:
        form = UpdateUploadImageForm(request.POST or None, request.FILES or None, instance=uploaded_image)
        if form.is_valid():
            obj = form.save(commit=False)
            
            # If a new image is uploaded, update the prediction
            if 'image' in request.FILES:
                image = Image.open(request.FILES['image'])
                predicted_class = predict_image(image)
                obj.result = f"Predicted class: {predicted_class}"
            
            obj.save()
            context['success_message'] = "Updated"
            uploaded_image = obj
            
    form = UpdateUploadImageForm(
        initial = {
            "title": uploaded_image.title,
            "result": uploaded_image.result,
            "image": uploaded_image.image,
        }
    )

    context['form'] = form
    return render(request, 'image_recognition/edit_image.html', context)

def upload_history_view(request):
    # Get sort parameter from URL
    sort_param = request.GET.get('sort')
    filter_param = request.GET.get('filter')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    page = request.GET.get('page', 1)  # Get the page number from request
    
    # Start with base queryset using select_related to reduce database queries for faster loading of page.
    queryset = UploadedImage.objects.select_related('author')
    
    # Apply date range filtering if provided
    if start_date and end_date:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            # Add one day to end_date to include the entire end date
            end_date = end_date + timedelta(days=1)
            queryset = queryset.filter(date_updated__range=(start_date, end_date))
        except ValueError:
            # Handle invalid date format
            pass

    # Apply filtering based on result parameter
    if filter_param:
        if filter_param == 'mange':
            queryset = queryset.filter(result__icontains='mange')
        elif filter_param == 'normal':
            queryset = queryset.filter(result__icontains='normal')

    # Apply sorting based on parameter
    if sort_param:
        if sort_param == 'default':
            # Default sorting
            queryset = queryset.order_by('-date_updated')
        elif sort_param == 'result':
            queryset = queryset.order_by('result')
        elif sort_param == '-result':
            queryset = queryset.order_by('-result')
        elif sort_param == 'author':
            queryset = queryset.order_by('author__username')  # Assuming author has username field
        elif sort_param == '-author':
            queryset = queryset.order_by('-author__username')
        elif sort_param == 'date_uploaded':
            queryset = queryset.order_by('date_updated')
        elif sort_param == '-date_uploaded':
            queryset = queryset.order_by('-date_updated')
    else:
        # Default sorting if no sort parameter
        queryset = queryset.order_by('-date_updated')

    # Create paginator instance
    paginator = Paginator(queryset, 10)

    try:
        uploaded_images = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        uploaded_images = paginator.page(1)
    except EmptyPage:
        # If page is out of range, deliver last page of results.
        uploaded_images = paginator.page(paginator.num_pages)
    
    context = {
        'uploaded_image': uploaded_images,
        'sort': sort_param,  # Pass current sort to template
        'filter': filter_param,  # Pass current filter to template
        'current_sort': sort_param if sort_param else 'default',  # For highlighting current sort option
        'current_filter': filter_param if filter_param else 'all',  # For highlighting current filter option
        'start_date': start_date if start_date else None,
        'end_date': end_date if end_date else None,
        'today': timezone.now().date(),  # For max date in date inputs
    }
    
    return render(request, 'image_recognition/upload_history.html', context)