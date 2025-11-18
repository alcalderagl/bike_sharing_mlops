from fastapi import APIRouter
router = APIRouter()

@router.get('/live')
def live():
    return {'status': 'ok'}

@router.get('/ready')
def live():
    return {'status': 'ready'}