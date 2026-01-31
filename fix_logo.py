#!/usr/bin/env python3
"""Fix the DRISHTI logo in app_main.py"""

import re

file_path = r'D:\Certifications\Hackathons\SnowHack\NoiseFloor\Noise_floor_project\dashboard\app_main.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the SVG eye logo
eye_svg = '''<svg width="44" height="44" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="eyeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#06b6d4"/>
                <stop offset="100%" style="stop-color:#0891b2"/>
            </linearGradient>
            <linearGradient id="irisGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#22d3ee"/>
                <stop offset="50%" style="stop-color:#06b6d4"/>
                <stop offset="100%" style="stop-color:#0e7490"/>
            </linearGradient>
            <filter id="glow"><feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        <ellipse cx="50" cy="50" rx="45" ry="28" fill="none" stroke="url(#eyeGrad)" stroke-width="4" filter="url(#glow)"/>
        <circle cx="50" cy="50" r="22" fill="url(#irisGrad)" filter="url(#glow)"/>
        <circle cx="50" cy="50" r="10" fill="#0f172a"/>
        <circle cx="44" cy="44" r="4" fill="rgba(255,255,255,0.7)"/>
        <path d="M5 50 Q50 20 95 50" fill="none" stroke="url(#eyeGrad)" stroke-width="3" opacity="0.6"/>
        <path d="M5 50 Q50 80 95 50" fill="none" stroke="url(#eyeGrad)" stroke-width="3" opacity="0.6"/>
    </svg>'''

# Find and replace the header section
old_pattern = r'''    status_class = latest_zone\.lower\(\) if latest_zone != 'STANDBY' else 'normal'
    
    # Header
    st\.markdown\(f"""
    <div class="main-header">
        <div class="logo-section">
            <div class="logo-icon">[^<]*</div>'''

new_header = f'''    status_class = latest_zone.lower() if latest_zone != 'STANDBY' else 'normal'
    
    # Eye SVG Logo for DRISHTI
    eye_svg = \'\'\'{eye_svg}\'\'\'
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <div class="logo-section">
            <div class="logo-icon">{{eye_svg}}</div>'''

content = re.sub(old_pattern, new_header, content)

# Clean up remaining corrupted characters  
content = content.replace('ðŸ"¹', '')
content = content.replace('ðŸ"¬', '')
content = content.replace('ðŸš', '')
content = content.replace('ðŸ"¡', '')
content = content.replace('ðŸ"Š', '')
content = content.replace('ðŸ§ ', '')
content = content.replace('ðŸš¨', '')
content = content.replace('âš™ï¸', '')

# Fix data mode display
content = content.replace('" Drone-Bird"', '"Drone-Bird"')
content = content.replace(' UCSD"', 'UCSD"')
content = content.replace(' Synthetic"', 'Synthetic"')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Logo fixed successfully!")
