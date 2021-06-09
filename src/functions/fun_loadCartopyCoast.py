import cartopy.feature as cfeature

def loadCartopyCoast():
    Coast = cfeature.NaturalEarthFeature(category='physical',scale='50m',
                                    facecolor='none', name='coastline')
    return Coast