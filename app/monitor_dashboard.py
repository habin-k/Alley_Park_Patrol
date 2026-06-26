from django.shortcuts import render, redirect
from django.contrib import messages

VALID_USERNAME = 'user'
VALID_PASSWORD = 'password'


def login_view(request):
    if request.session.get('username'):
        return redirect('monitor_dashboard')
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            request.session['username'] = username
            return redirect('monitor_dashboard')
        messages.error(request, '아이디 또는 비밀번호가 올바르지 않습니다.')
    return render(request, 'login_center.html')


def logout_view(request):
    request.session.flush()
    return redirect('login')


def dashboard(request):
    if not request.session.get('username'):
        return redirect('login')
    return render(request, 'monitor_dashboard.html', {
        'username': request.session['username']
    })
